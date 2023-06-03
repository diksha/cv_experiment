from abc import ABC
from typing import Any, Callable, List, Union, final

import boto3
from pyflink.common import JobExecutionResult
from pyflink.common.typeinfo import Types
from pyflink.datastream import (
    FlatMapFunction,
    KeySelector,
    RuntimeExecutionMode,
    StreamExecutionEnvironment,
    WindowedStream,
)
from pyflink.datastream.connectors import Sink
from pyflink.datastream.window import TumblingProcessingTimeWindows

from protos.platform.bowser.v1.bowser_config_consumer_pb2 import (
    ProcessorConsumer,
)
from protos.platform.bowser.v1.bowser_env_config_pb2 import BowserEnv
from services.platform.bowser.engine.common.flatmaps.aws_bucket_to_s3_object_flat_map import (
    AwsBucketToS3ObjectFlatMap,
)
from services.platform.bowser.engine.common.flatmaps.aws_s3_object_to_states_events_flat_map import (
    AwsS3ObjectToStatesEventsFlatMap,
)
from services.platform.bowser.engine.common.reducer.count_keys import CountKeys

# trunk-ignore-all(pylint/E0611,pylint/C0301,pylint/W9012,pylint/W0108): ignore pb import errors


class BowserFactoryPipe:
    @staticmethod
    def new_raw_pipe(
        mode: RuntimeExecutionMode, max_parallelism: int
    ) -> "BowserPipe":
        """The new_raw_pipe function is a factory function that creates a new BowserPipe object.

        Args:
            mode: RuntimeExecutionMode: Specify the execution mode of the pipe
            max_parallelism: int: Specify the maximum number of

        Returns:
            A bowserpipe

        """
        return BowserPipe(mode, max_parallelism)

    @staticmethod
    def new_config_pipe(bowser_env: BowserEnv) -> "BowserPipe":
        """The new_config_pipe function is a factory function that creates a BowserPipe object.
        The BowserPipe object is the main entry point for using bowser.  It provides methods to
        create new pipelines, and it also provides methods to execute those pipelines in various ways.

        Args:
            bowser_env: BowserEnv: Pass the environment variables

        Returns:
            A bowserpipe object
        """
        return BowserPipe(
            RuntimeExecutionMode(bowser_env.mode), bowser_env.max_parallelism
        )


class BowserPipe(ABC):
    def __init__(self, mode: RuntimeExecutionMode, max_parallelism: int):
        """The __init__ function is called when the class is instantiated.
        It sets up the environment for running a Flink job, and initializes an AWS object to be used later.

        Args:
            self: Represent the instance of the class
            mode: RuntimeExecutionMode: Set the runtime mode of the flink cluster
            max_parallelism: int: Set the maximum parallelism of the flink job

        Returns:
            Nothing
        """
        self.pipe = StreamExecutionEnvironment.get_execution_environment()
        self.pipe.set_max_parallelism(max_parallelism)
        self.pipe.set_runtime_mode(mode)
        self.__aws = None

    def consume_states_event_from_s3(
        self, s3_config: ProcessorConsumer, source_name: str
    ) -> "BowserFlow":
        """The consume_states_event_from_s3 function is a helper function that consumes states and events from S3.
        It takes in a s3_config object, which contains the information needed to process S3 files, as well as a source_name
        string. The source name string is used for logging/metric purposes only. It returns a BowserFlow object that can be used
        to create jobs with.

        Args:
            self: Refer to the object itself
            s3_config: ProcessorConsumer: Specify the s3 bucket and prefix to read from
            source_name: str: Identify the name of the source for metric/logging purpose

        Returns:
            A bowserflow object

        """
        return (
            self.consume_from_collection(s3_config, source_name)
            .flat_map(
                AwsBucketToS3ObjectFlatMap(self.__aws_session()),
                "[FLATMAP] LIST STATES/EVENT S3 FILE",
            )
            .key_by(lambda x: "/".join(x[1].split("/")[1:5]))
            .flat_map(
                AwsS3ObjectToStatesEventsFlatMap(self.__aws_session()),
                "[FLATMAP] PROTO SERIALIZATION STATES/EVENT",
            )
        )

    def consume_from_collection(
        self, collection: List[Any], source_name: str
    ) -> "BowserFlow":
        """
        The consume_from_collection function is a wrapper around the pipe.from_collection function,
        which takes in a list of objects and returns an instance of BowserFlow. The BowserFlow class
        is used to keep track of the current state of the pipeline, as well as provide methods for
        adding new stages to it.

        Args:
            self: Refer to the current instance of a class
            collection: List[Any]: Specify the collection of data to be consumed
            source_name: str: Name the source of the data

        Returns:
            A bowserflow object

        """
        bowser_flow = self.pipe.from_collection(
            collection,
        ).name(source_name)
        return BowserFlow(bowser_flow)

    @final
    def __aws_session(self):
        """The __aws_session function is a private function that creates an AWS session.
        It uses the boto3 library to create a session object, which can be used to access
        AWS resources. The __aws_session function is called by other functions in this class.

        Args:
            self: Represent the instance of the class

        Returns:
            The boto3 session

        """
        if self.__aws is None:
            self.__aws = boto3.Session()
        return self.__aws


class BowserFlow(ABC):
    def __init__(self, original_stream):
        """The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines what will be done with it.
        In this case, we are setting up a new DataStream object that will be used to
        read data from a BowserPipe object.

        Args:
            self: Represent the instance of the class
            pipe: BowserPipe: Store the pipe that is passed in
            original_stream: DataStream or DataKeyedStream: Store the original stream

        Returns:
            Nothing

        """
        self._original_stream = original_stream

    def flat_map(
        self, function: FlatMapFunction, function_name: str
    ) -> "BowserFlow":
        """The flat_map function is a combination of the map and flat_map functions.
        It applies a function to each element in the stream, then flattens all of
        the resulting elements into one stream. This is useful for when you want to
        apply an operation that returns multiple values per input value.

        Args:
            self: Refer to the current instance of the class
            function: FlatMapFunction: Transform the input stream into a new output stream
            function_name: str: Name the function

        Returns:
            A bowserflow object

        """
        flat_map = self._original_stream.flat_map(function)
        flat_map.name(function_name)
        return BowserFlow(flat_map)

    def key_by(
        self, key_selector: Union[Callable, KeySelector]
    ) -> "BowserFlow":
        """The key_by function is used to partition the stream by a key.
        The function returns a KeyedStream where each item of the original stream is assigned to one or more keys using
        a user-defined function. The items with same key are guaranteed to be in the same partition (and processed by
        the same task manager). This method can be used for grouping, aggregation and windowing operations.

        Args:
            self: Represent the instance of the class
            key_selector: Union[Callable: Specify the key_selector function
            KeySelector]: Specify the key that is used to group

        Returns:
            A bowserkeyflow object
        """
        key_steam = self._original_stream.key_by(key_selector)
        return BowserFlow(key_steam)

    def key_by_incident_with_window(
        self,
        key_reducer: FlatMapFunction,
        window: TumblingProcessingTimeWindows,
    ) -> "BowserAggregator":
        """The key_by_incident_with_window function is used to create a keyed stream from an incident stream.
        The function takes two arguments:
            1) A FlatMapFunction that maps the incident to a tuple of (key, value). The key will be used as the
               partitioning key for the window and should be unique per incident. The value can be any type and
               will become part of the output tuple when aggregating over windows. For example, if you wanted to
               count incidents by their severity level, you could use this function:

        Args:
            self: Refer to the instance of the class
            key_reducer: FlatMapFunction: Create a tuple of the incident and the key
            window: TumblingProcessingTimeWindows: Assign the window to the stream
            : Assign a window to each key

        Returns:
            A bowseraggregator object

        """
        flat_map_keyed = self.flat_map(
            key_reducer, "[FLATMAP] INCIDENT TO TUPLE[KEYS]"
        )
        key_steam = flat_map_keyed.key_by(lambda x: x)
        window_stream = key_steam.get_original_stream().window(window)
        return BowserAggregator(window_stream)

    def sink_and_submit_job(
        self, sink: Sink, sink_name: str
    ) -> JobExecutionResult:
        """The sink_to function is used to send the output of a stream to a sink.

        Args:
            self: Refer to the current object
            sink: Sink: Specify the sink to which the stream is being sent
            sink_name: str: Name the sink

        Returns:
            The pipeline, which is the flink JobExecutionResult
        """
        self._original_stream.sink_to(sink).name(sink_name)
        return self._original_stream.get_execution_environment().execute()

    def get_original_stream(self):
        """The get_original_stream function returns the original stream that was passed to the constructor.

        Args:
            self: Represent the instance of the class

        Returns:
            The original stream of the object

        Doc Author:
            Trelent
        """
        return self._original_stream

    def to_string(self):
        """The to_string function converts the input stream to a string.

        Args:
            self: Access the attributes and methods of the class

        Returns:
            A bowserflow object

        """
        to_string = self._original_stream.map(
            lambda x: str(x), Types.STRING()
        ).name("[FLATMAP] OBJECT TO STRING")
        return BowserFlow(to_string)

    def submit_job(self, job_name: str) -> JobExecutionResult:
        """
        The submit_job function

        Args:
            self: Represent the instance of the class
            job_name: str: Specify the name of the job to be submitted

        Returns:
            A jobexecutionresult object

        """
        return self._original_stream.get_execution_environment().execute(
            job_name
        )


class BowserAggregator(ABC):
    def __init__(self, stream: WindowedStream):
        """The __init__ function is called when the object is created.
        It initializes the object with a BowserPipe and WindowedStream.
        The BowserPipe will be used to send data to the browser, while
        the WindowedStream will be used as a source of data.

        Args:
            self: Represent the instance of the class
            pipe: BowserPipe: Access the pipe that is used to send data to the next stage of the pipeline
            stream: WindowedStream: Pass the original stream to the class

        Returns:
            An instance of the class

        """
        self._original_stream = stream

    def count_incident_keys(self):
        """The count_incident_keys function counts the number of keys in a stream.

        Args:
            self: Bind the method to an object

        Returns:
            A bowserflow object

        """
        incident_stream = self._original_stream.aggregate(
            aggregate_function=CountKeys()
        )
        return BowserFlow(incident_stream)
