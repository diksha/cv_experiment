from os import makedirs, mkdir
from os.path import exists
from pathlib import Path
from shutil import rmtree
from sys import getsizeof

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from neo4j.spatial import Point
from neo4j.time import Date, DateTime, Duration, Time
from tqdm import tqdm

from third_party.neo4j_backup.backends import get_unique_prop_key, to_json


# trunk-ignore-all(flake8/B008,flake8/D201,pylint/C0116)
# trunk-ignore-all(pylint/C0301,pylint/W1309,flake8/F541)
# trunk-ignore-all(pylint/W0707,pylint/W9011,pylint/W9012)
# trunk-ignore-all(pylint/R0912,flake8/W293,pylint/W0612)
class Extractor:
    def __init__(
        self,
        project_dir,
        driver: GraphDatabase.driver,
        database: str = "neo4j",
        input_yes: bool = False,
        compress: bool = True,
        json_file_size: int = int("0xFFFF", 16),
    ):

        """
        The purpose of this class is to extract all the information from a neo4j graph

        :param project_dir: The directory where to back up Neo4j Graph
        :param driver: Neo4j driver
        :param input_yes: bool, determines weather to just type in "y" for all input options
        :param compress: bool, weather or not to compress files as they are being extracted
        :param json_file_size: int, max size of json object in memory before dumping
        """
        # voxel changes start: don't use cwd for project dir (bazel)
        self.project_dir: Path = Path(project_dir)
        self.data_dir: Path = self.project_dir / "data"
        # voxel changes end
        self.driver: GraphDatabase.driver = driver
        self.database: str = database
        self.input_yes: bool = input_yes
        self.compress: bool = compress

        self.property_keys: set = set()
        self.labels: set = set()
        self.rel_types: set = set()
        self.constraints: list = []
        self.constraints_names: list = []
        self.db_id: str = ""

        self.json_file_size: int = (
            json_file_size  # Default size of json objects in memory
        )

    def extract_data(self):

        self._test_connection()
        self._verify_db_not_empty()

        self._pull_db_id()  # Get ID of database

        if exists(self.project_dir):

            if self.input_yes:
                rmtree(self.project_dir)

            else:
                user_input = input(
                    f"The directory {self.project_dir} already exist, would you like to replace the "
                    f"directory? (y/N)\n"
                )
                if user_input.lower() == "y":
                    rmtree(self.project_dir)
                else:
                    raise UserWarning(
                        "Aborted, project_dir directory already exists"
                    )
        # voxel changes start: don't use cwd for project dir (bazel)
        if not exists(self.project_dir):
            makedirs(self.project_dir, exist_ok=False)
        mkdir(self.data_dir)
        # voxel changes end
        self._pull_constraints()  # get constraints of database
        self._pull_nodes()  # get nodes in database
        self._pull_relationships()  # get relationship in database

        # calculate a unique prop key to act a dummy id prop for importing
        unique_prop_key = self._calc_unique_prop_key()

        # Store meta data
        to_json(file_path=self.project_dir / f"db_id.json", data=self.db_id)
        to_json(
            file_path=self.project_dir / f"unique_prop_key.json",
            data=unique_prop_key,
        )
        to_json(
            file_path=self.project_dir / f"constraints.json",
            data=self.constraints,
        )
        to_json(
            file_path=self.project_dir / f"constraints_names.json",
            data=self.constraints_names,
        )
        to_json(
            file_path=self.project_dir / "property_keys.json",
            data=list(self.property_keys),
        )
        to_json(
            file_path=self.project_dir / "node_labels.json",
            data=list(self.labels),
        )
        to_json(
            file_path=self.project_dir / "rel_types.json",
            data=list(self.rel_types),
        )
        to_json(
            file_path=self.project_dir / "compressed.json", data=self.compress
        )

    def _test_connection(self):
        try:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (a) RETURN a LIMIT 1")
        except ServiceUnavailable:
            raise ServiceUnavailable(
                "Unable to connect to database. If this is a local database, make sure the "
                "database is running. If this is a remote database, make sure the correct "
                "database is referenced."
            )

    def _verify_db_not_empty(self):
        with self.driver.session(database=self.database) as session:
            results = session.run("MATCH (a) RETURN a LIMIT 1").data()
            if not results:
                raise LookupError(
                    "There is not data to pull from the database, make sure the correct database is "
                    "referenced/running."
                )

    def _pull_db_id(self):

        with self.driver.session(database=self.database) as session:
            results = session.run("CALL db.info")
            for result in results:
                self.db_id = dict(result)["id"]

    def _pull_constraints(self):

        with self.driver.session(database=self.database) as session:
            # voxel changes start: neo4j 5.0 constraint backup
            results = session.run(
                "SHOW CONSTRAINT YIELD name, createStatement"
            )
            for result in results:
                name = dict(result)["name"]
                raw = dict(result)["createStatement"]
                components = (
                    raw.split(" OPTIONS")[0].replace("`", "").split(name)
                )
                constraint = (
                    components[0] + f"{name} IF NOT EXISTS" + components[1]
                )
                self.constraints.append(constraint)
                self.constraints_names.append(name)
            # voxel changes end

    @staticmethod
    def __parse_props_types(props):
        def __parse_prop(prop):

            if isinstance(prop, bool):
                data_type = "bool"
            elif isinstance(prop, float):
                data_type = "float"
            elif isinstance(prop, int):
                data_type = "int"
            elif isinstance(prop, str):
                data_type = "str"
            elif isinstance(prop, Point):
                point_srid = prop.srid
                if point_srid == 7203:
                    data_type = "2d-cartesian-point"
                elif point_srid == 9157:
                    data_type = "3d-cartesian-point"
                elif point_srid == 4326:
                    data_type = "2d-WGS-84-point"
                elif point_srid == 4979:
                    data_type = "3d-WGS-84-point"
                else:
                    raise ValueError(
                        f"Point of srid {point_srid} is not supported"
                    )
                prop = list(prop)
            elif isinstance(prop, Date):
                data_type = "date"
            elif isinstance(prop, Time):
                data_type = "time"
            elif isinstance(prop, DateTime):
                data_type = "datetime"
            elif isinstance(prop, Duration):
                data_type = "duration"
                time_list = ["months", "days", "seconds", "nanoseconds"]
                prop = dict(zip(time_list, props[prop_key]))
            else:
                raise ValueError(
                    f"Encoder is not setup for {type(prop)} type from {prop} on "
                    f"prop key {prop_key}"
                )
            return data_type, prop

        node_props_types = {}
        for prop_key, prop_value in props.items():

            if isinstance(prop_value, list):
                prop_values = []
                prop_types = []
                for sub_prop_value in prop_value:
                    prop_type, prop_value = __parse_prop(sub_prop_value)
                    prop_types.append(prop_type)
                    prop_values.append(prop_value)
                props[prop_key] = prop_values
                node_props_types[prop_key] = prop_types

            else:
                prop_type, prop_value = __parse_prop(prop_value)
                props[prop_key] = prop_value
                node_props_types[prop_key] = prop_type

        return props, node_props_types

    def __parse_node__(self, node):
        node_id = node.id
        node_labels = list(node.labels)
        node_props = dict(node)
        node_props, node_props_types = self.__parse_props_types(node_props)

        return node_id, node_labels, node_props, node_props_types

    def _pull_nodes(self):

        query = """
        
        MATCH (node)
        RETURN node
        
        """

        extracted_data = []

        with self.driver.session(database=self.database) as session:
            number_of_nodes = session.run(
                "MATCH (node) RETURN COUNT(node)"
            ).value()[0]
            results = session.run(query)

            counter = 0
            for record in tqdm(
                results, total=number_of_nodes, desc="Extracting Nodes"
            ):
                counter += 1

                # Base node object
                node = record["node"]
                (
                    node_id,
                    node_labels,
                    node_props,
                    node_props_types,
                ) = self.__parse_node__(node)
                self.property_keys.update(node_props.keys())
                self.labels.update(node_labels)

                row = {
                    "node_id": node_id,
                    "node_labels": node_labels,
                    "node_props": node_props,
                    "node_props_types": node_props_types,
                }
                extracted_data.append(row)

                size_in_ram = getsizeof(extracted_data)
                if size_in_ram > self.json_file_size:
                    to_json(
                        self.data_dir / f"nodes_{counter}.json",
                        extracted_data,
                        compress=self.compress,
                    )
                    extracted_data = []

            # dump and compress remaining data
            if extracted_data:
                to_json(
                    self.data_dir / f"nodes_{counter}.json",
                    extracted_data,
                    compress=self.compress,
                )

    def _pull_relationships(self):

        query = """

        MATCH (start_node)-[rel]->(end_node)
        RETURN start_node, end_node, rel

        """

        extracted_data = []

        with self.driver.session(database=self.database) as session:
            number_of_relationships = session.run(
                "MATCH p=(start_node)-[rel]->(end_node) RETURN COUNT(p)"
            ).value()[0]
            results = session.run(query)

            counter = 0
            for record in tqdm(
                results,
                total=number_of_relationships,
                desc="Extracting Relationships",
            ):
                counter += 1

                # Gather starting_node
                start_node = record["start_node"]
                (
                    start_node_id,
                    start_node_labels,
                    start_node_props,
                    start_node_props_types,
                ) = self.__parse_node__(start_node)
                self.property_keys.update(start_node_props.keys())
                self.labels.update(start_node_labels)

                # Gather ending_node
                end_node = record["end_node"]
                (
                    end_node_id,
                    end_node_labels,
                    end_node_props,
                    end_node_props_types,
                ) = self.__parse_node__(end_node)
                self.property_keys.update(end_node_props.keys())
                self.labels.update(end_node_labels)

                # Gather relationship
                rel = record["rel"]
                rel_type = rel.type
                rel_props = dict(rel)
                rel_props, rel_props_types = self.__parse_props_types(
                    rel_props
                )
                self.property_keys.update(rel_props.keys())
                self.rel_types.add(rel_type)

                row = {
                    "start_node_id": start_node_id,
                    "start_node_labels": start_node_labels,
                    "end_node_id": end_node_id,
                    "end_node_labels": end_node_labels,
                    "rel_type": rel_type,
                    "rel_props": rel_props,
                    "rel_props_types": rel_props_types,
                }
                extracted_data.append(row)

                size_in_ram = getsizeof(extracted_data)
                if size_in_ram > self.json_file_size:
                    to_json(
                        self.data_dir / f"relationships_{counter}.json",
                        extracted_data,
                        compress=self.compress,
                    )
                    extracted_data = []

            # dump and compress remaining data
            if extracted_data:
                to_json(
                    self.data_dir / f"relationships_{counter}.json",
                    extracted_data,
                    compress=self.compress,
                )

    def _calc_unique_prop_key(self):
        keys_to_avoid = self.property_keys.copy()
        keys_to_avoid.update(self.constraints_names)

        # Neo4j's built in IDs can change as new entities are added. So, a unique property is generated where the
        # pulled ids are placed temporarily. A unique property is calculated because we do not want to 'create' a
        # dummy property that the user actually uses.
        unique_prop_key = get_unique_prop_key(keys_to_avoid)
        return unique_prop_key
