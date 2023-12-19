#!/bin/bash

set -e
set -x

DB_NAME="ssb_grid_tmp"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$DIR")"


function pexec() {
  echo "$*" | time psql -X -v ON_ERROR_STOP=1 "ssb_grid_tmp"
}

function create_ssb_db() {
	dropdb --if-exists "$DB_NAME"
	createdb "$DB_NAME"
	pexec "CREATE EXTENSION postgis;"
}

function main() {
	create_ssb_db

	# # https://kartkatalog.geonorge.no/metadata/befolkning-paa-rutenett-1000-m-2019/fab7c42f-9eb1-4eab-8984-ffd744c86343

	pop_grid_file="Befolkning_0000_Norge_25833_BefolkningsstatistikkRutenett1km2019_GML"

	if [[ ! -f "$pop_grid_file.gml" ]];
	then
		curl -O https://nedlasting.geonorge.no/geonorge/Befolkning/BefolkningsstatistikkRutenett1km2019/GML/Befolkning_0000_Norge_25833_BefolkningsstatistikkRutenett1km2019_GML.zip
		unzip "$pop_grid_file.zip"
	fi

	ogr2ogr \
		-f postgresql \
		"PG:host=localhost dbname=$DB_NAME" \
		"$pop_grid_file.gml" \
		"BefolkningPåRuter1km" \
		-nln befolkning1x1 \
		-forceNullable \
		--config PG_USE_COPY YES


	ogr2ogr \
		-f postgresql \
		"PG:host=localhost dbname=${DB_NAME}" \
		"WFS:https://wfs.geonorge.no/skwms1/wfs.administrative_enheter?&service=WFS&acceptversions=2.0.0&request=GetCapabilities" \
		app:Kommune \
		-nln kommuner \
		-forceNullable \
		--config OGR_WFS_PAGING_ALLOWED ON \
		--config OGR_WFS_PAGE_SIZE 1000 \
		--config PG_USE_COPY YES \
		-makevalid \
		-nlt PROMOTE_TO_MULTI

	pexec "
		DROP TABLE IF EXISTS pop_grid_fixed;

		CREATE TABLE pop_grid_fixed AS
		SELECT
			befolkning1x1.poptot as population,
			befolkning1x1.ssbid1000m as ssbid,
			LPAD(kommunenummer::text, 4, '0') as kommunekode,
			navn[1] as kommunenavn,
			befolkning1x1.område as geom
		FROM befolkning1x1
		INNER JOIN kommuner ON st_intersects(st_transform(kommuner.område, 25833), befolkning1x1.område)
		WHERE poptot is not null;
	"

	# consider only cells that are fully within a municipality
	pexec "
		DROP TABLE IF EXISTS muni_cells;
		CREATE TABLE muni_cells AS
			WITH a AS (
				SELECT
					*,
					count(*) OVER (PARTITION BY ssbid) AS muni_count
				FROM
					pop_grid_fixed
			)
			SELECT
				*,
				/* use ROW_NUMBER instead of RANK to avoid duplicates when two cells are tied */
				ROW_NUMBER() OVER (PARTITION BY kommunekode ORDER BY population DESC) AS rank
			FROM
				a
			WHERE
				muni_count = 1
	"

	pexec "
		DROP TABLE IF EXISTS central_cells;
		CREATE TABLE central_cells AS
		SELECT * FROM muni_cells WHERE rank = 1
	"

	pexec "
	DROP TABLE IF EXISTS central_points;
	CREATE TABLE central_points AS
		SELECT
			population,
			ssbid,
			kommunekode,
			kommunenavn,
			ST_Centroid(geom) as geom,
			'centroid' as point_type,
			NULL::int as distance,
			NULL::int as bearing
		FROM central_cells;
	"

	for distance in "5000 5km" "10000 10km" "20000 20km"
	do
	    IFS=" " set -- $distance
		distance_meter="$1"
		distance_desc="$2"

		for bearing in "0 right" "90 top" "270 bottom" "180 left"
		do
			IFS=" " set -- $bearing
			bearing_degrees="$1"
			bearing_desc="$2"

			pexec "
				INSERT INTO central_points (kommunekode, kommunenavn, geom, point_type, distance, bearing)
				SELECT
					kommunekode,
					kommunenavn,
					ST_SetSRID(
						ST_Translate(
							ST_Rotate(
								ST_MakePoint(${distance_meter}, 0.0),
								radians(${bearing_degrees})
							),
							ST_X(geom),
							ST_Y(geom)
						),
						ST_SRID(geom)
					) AS geom,
					'${distance_desc}:${bearing_desc}' as point_type,
					${distance_meter} as distance,
					${bearing_degrees} as bearing
				FROM central_points where point_type = 'centroid';
			"
		done
	done

	mkdir -p "${PROJECT_DIR}/data/geo"

	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/storste-km2-per-kommune-utm33.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-sql "select population, ssbid, kommunekode, kommunenavn, geom from central_cells"

	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/muni-cells-utm33.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-sql "select population, ssbid, kommunekode, kommunenavn, rank, geom from muni_cells"

	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/kommuner-utm33.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-sql "select LPAD(kommunenummer::text, 4, '0') as kommunekode, navn as kommunenavn, st_transform(område, 25833) as geom from kommuner"


	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/storste-km2-per-kommune-4326.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-sql "select population, ssbid, kommunekode, kommunenavn, st_transform(geom, 4326) as geom from central_cells"

	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/central-points-utm33.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-sql "select kommunekode, kommunenavn, point_type, distance, bearing, geom from central_points"

	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/central-points-4326.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-sql "select kommunekode, kommunenavn, point_type, distance, bearing, st_transform(geom, 4326) as geom from central_points"
}


main
