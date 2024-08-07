#!/bin/bash

set -e
set -x

POP_YEAR="2023"
DB_NAME="ssb_grid_tmp_${POP_YEAR}"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$DIR")"

function pexec() {
  echo "$*" | time psql -X -v ON_ERROR_STOP=1 "$DB_NAME"
}

function create_ssb_db() {
	dropdb --if-exists "$DB_NAME"
	createdb "$DB_NAME"
	pexec "CREATE EXTENSION postgis;"
}

function main() {
	create_ssb_db

	# # https://kartkatalog.geonorge.no/metadata/befolkning-paa-rutenett-1000-m-2019/fab7c42f-9eb1-4eab-8984-ffd744c86343

	local pop_grid_file="befolkning_2023"
	local layer_name="layer_337"

	if [[ ! -f "$pop_grid_file.gml" ]];
	then
		curl -o "$pop_grid_file.gml" "https://ogc.ssb.no/wms.ashx?service=WFS&version=1.1.0&request=GetFeature&typename=layer_337&outputformat=GML3"
	fi

	ogr2ogr \
		-f postgresql \
		"PG:host=localhost dbname=$DB_NAME" \
		"$pop_grid_file.gml" \
		"$layer_name" \
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
		-nlt PROMOTE_TO_MULTI \
		-skipfailures

	ogr2ogr \
		-f postgresql \
		"PG:host=localhost dbname=${DB_NAME}" \
		"WFS:https://wfs.geonorge.no/skwms1/wfs.administrative_enheter?&service=WFS&acceptversions=2.0.0&request=GetCapabilities" \
		app:Fylke \
		-nln fylker \
		-forceNullable \
		--config OGR_WFS_PAGING_ALLOWED ON \
		--config OGR_WFS_PAGE_SIZE 1000 \
		--config PG_USE_COPY YES \
		-makevalid \
		-nlt PROMOTE_TO_MULTI \
		-skipfailures


	pexec "
		DROP TABLE IF EXISTS pop_grid_fixed;

		CREATE TABLE pop_grid_fixed AS
		SELECT
			befolkning1x1.pop_tot as population,
			befolkning1x1.pop_ave2 as average_age,
			befolkning1x1.pop_mal as population_male,
			befolkning1x1.pop_fem as population_female,
			befolkning1x1.ssbid_1000m as ssbid,
			LPAD(kommunenummer::text, 4, '0') as kommunekode,
			befolkning1x1.msgeometry as geom,
			CASE
				WHEN kommuner.språk[1] = 'nor' THEN navn[1]
				WHEN kommuner.språk[2] = 'nor' THEN navn[2]
				ELSE navn[1]
			END as kommunenavn
		FROM befolkning1x1
		INNER JOIN kommuner ON st_intersects(st_transform(kommuner.område, 32633), befolkning1x1.msgeometry)
		WHERE befolkning1x1.pop_tot is not null;
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

	coordinate_precision=5

	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/storste-km2-per-kommune-utm33.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-lco COORDINATE_PRECISION=${coordinate_precision} \
		-sql "select population, ssbid, kommunekode, kommunenavn, st_transform(geom, 25833) as geom from central_cells"

	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/muni-cells-utm33.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-lco COORDINATE_PRECISION=${coordinate_precision} \
		-sql "select population, ssbid, kommunekode, kommunenavn, rank, st_transform(geom, 25833) as geom from muni_cells"


	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/top-muni-points-4326.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-lco COORDINATE_PRECISION=${coordinate_precision} \
		-sql "select population, ssbid, kommunekode, kommunenavn, rank, st_transform(st_centroid(geom), 4326) as geom from muni_cells where rank <= 10"

	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/kommuner-utm33.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-lco COORDINATE_PRECISION=${coordinate_precision} \
		-sql "select LPAD(kommunenummer::text, 4, '0') as kommunekode, navn as kommunenavn, st_transform(område, 25833) as geom from kommuner"


	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/storste-km2-per-kommune-4326.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-lco COORDINATE_PRECISION=${coordinate_precision} \
		-sql "select population, ssbid, kommunekode, kommunenavn, st_transform(geom, 4326) as geom from central_cells"

	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/central-points-utm33.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-lco COORDINATE_PRECISION=${coordinate_precision} \
		-sql "select kommunekode, kommunenavn, point_type, distance, bearing, st_transform(geom, 25833) as geom from central_points"

	ogr2ogr \
		-f GeoJSON \
		"${PROJECT_DIR}/data/geo/central-points-4326.geojson" \
		"PG:host=localhost dbname=${DB_NAME}" \
		-lco COORDINATE_PRECISION=${coordinate_precision} \
		-sql "select kommunekode, kommunenavn, point_type, distance, bearing, st_transform(geom, 4326) as geom from central_points"
}


main
