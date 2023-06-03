// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        v3.20.3
// source: protos/platform/bowser/v1/bowser_config_consumer.proto

package v1

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type ProcessorConsumer struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Name string                `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Aws  *ProcessorConsumerAws `protobuf:"bytes,2,opt,name=aws,proto3" json:"aws,omitempty"`
}

func (x *ProcessorConsumer) Reset() {
	*x = ProcessorConsumer{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessorConsumer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessorConsumer) ProtoMessage() {}

func (x *ProcessorConsumer) ProtoReflect() protoreflect.Message {
	mi := &file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessorConsumer.ProtoReflect.Descriptor instead.
func (*ProcessorConsumer) Descriptor() ([]byte, []int) {
	return file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescGZIP(), []int{0}
}

func (x *ProcessorConsumer) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *ProcessorConsumer) GetAws() *ProcessorConsumerAws {
	if x != nil {
		return x.Aws
	}
	return nil
}

type ProcessorConsumerAws struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	S3      *ProcessorConsumerAwsS3      `protobuf:"bytes,1,opt,name=s3,proto3" json:"s3,omitempty"`
	Kinesis *ProcessorConsumerAwsKinesis `protobuf:"bytes,2,opt,name=kinesis,proto3" json:"kinesis,omitempty"`
}

func (x *ProcessorConsumerAws) Reset() {
	*x = ProcessorConsumerAws{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessorConsumerAws) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessorConsumerAws) ProtoMessage() {}

func (x *ProcessorConsumerAws) ProtoReflect() protoreflect.Message {
	mi := &file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessorConsumerAws.ProtoReflect.Descriptor instead.
func (*ProcessorConsumerAws) Descriptor() ([]byte, []int) {
	return file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescGZIP(), []int{1}
}

func (x *ProcessorConsumerAws) GetS3() *ProcessorConsumerAwsS3 {
	if x != nil {
		return x.S3
	}
	return nil
}

func (x *ProcessorConsumerAws) GetKinesis() *ProcessorConsumerAwsKinesis {
	if x != nil {
		return x.Kinesis
	}
	return nil
}

type ProcessorConsumerAwsKinesis struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *ProcessorConsumerAwsKinesis) Reset() {
	*x = ProcessorConsumerAwsKinesis{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessorConsumerAwsKinesis) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessorConsumerAwsKinesis) ProtoMessage() {}

func (x *ProcessorConsumerAwsKinesis) ProtoReflect() protoreflect.Message {
	mi := &file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessorConsumerAwsKinesis.ProtoReflect.Descriptor instead.
func (*ProcessorConsumerAwsKinesis) Descriptor() ([]byte, []int) {
	return file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescGZIP(), []int{2}
}

type ProcessorConsumerAwsS3 struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Buckets []*ProcessorConsumerAwsS3Bucket `protobuf:"bytes,1,rep,name=buckets,proto3" json:"buckets,omitempty"`
}

func (x *ProcessorConsumerAwsS3) Reset() {
	*x = ProcessorConsumerAwsS3{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessorConsumerAwsS3) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessorConsumerAwsS3) ProtoMessage() {}

func (x *ProcessorConsumerAwsS3) ProtoReflect() protoreflect.Message {
	mi := &file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessorConsumerAwsS3.ProtoReflect.Descriptor instead.
func (*ProcessorConsumerAwsS3) Descriptor() ([]byte, []int) {
	return file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescGZIP(), []int{3}
}

func (x *ProcessorConsumerAwsS3) GetBuckets() []*ProcessorConsumerAwsS3Bucket {
	if x != nil {
		return x.Buckets
	}
	return nil
}

type ProcessorConsumerAwsS3Bucket struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Name string   `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Uris []string `protobuf:"bytes,2,rep,name=uris,proto3" json:"uris,omitempty"`
}

func (x *ProcessorConsumerAwsS3Bucket) Reset() {
	*x = ProcessorConsumerAwsS3Bucket{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessorConsumerAwsS3Bucket) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessorConsumerAwsS3Bucket) ProtoMessage() {}

func (x *ProcessorConsumerAwsS3Bucket) ProtoReflect() protoreflect.Message {
	mi := &file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessorConsumerAwsS3Bucket.ProtoReflect.Descriptor instead.
func (*ProcessorConsumerAwsS3Bucket) Descriptor() ([]byte, []int) {
	return file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescGZIP(), []int{4}
}

func (x *ProcessorConsumerAwsS3Bucket) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *ProcessorConsumerAwsS3Bucket) GetUris() []string {
	if x != nil {
		return x.Uris
	}
	return nil
}

var File_protos_platform_bowser_v1_bowser_config_consumer_proto protoreflect.FileDescriptor

var file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDesc = []byte{
	0x0a, 0x36, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72,
	0x6d, 0x2f, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72, 0x2f, 0x76, 0x31, 0x2f, 0x62, 0x6f, 0x77, 0x73,
	0x65, 0x72, 0x5f, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x5f, 0x63, 0x6f, 0x6e, 0x73, 0x75, 0x6d,
	0x65, 0x72, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x19, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73,
	0x2e, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2e, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72,
	0x2e, 0x76, 0x31, 0x22, 0x6a, 0x0a, 0x11, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72,
	0x43, 0x6f, 0x6e, 0x73, 0x75, 0x6d, 0x65, 0x72, 0x12, 0x12, 0x0a, 0x04, 0x6e, 0x61, 0x6d, 0x65,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x41, 0x0a, 0x03,
	0x61, 0x77, 0x73, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x2f, 0x2e, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x73, 0x2e, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2e, 0x62, 0x6f, 0x77, 0x73,
	0x65, 0x72, 0x2e, 0x76, 0x31, 0x2e, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x43,
	0x6f, 0x6e, 0x73, 0x75, 0x6d, 0x65, 0x72, 0x41, 0x77, 0x73, 0x52, 0x03, 0x61, 0x77, 0x73, 0x22,
	0xab, 0x01, 0x0a, 0x14, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x43, 0x6f, 0x6e,
	0x73, 0x75, 0x6d, 0x65, 0x72, 0x41, 0x77, 0x73, 0x12, 0x41, 0x0a, 0x02, 0x73, 0x33, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x0b, 0x32, 0x31, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x70, 0x6c,
	0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2e, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72, 0x2e, 0x76, 0x31,
	0x2e, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x43, 0x6f, 0x6e, 0x73, 0x75, 0x6d,
	0x65, 0x72, 0x41, 0x77, 0x73, 0x53, 0x33, 0x52, 0x02, 0x73, 0x33, 0x12, 0x50, 0x0a, 0x07, 0x6b,
	0x69, 0x6e, 0x65, 0x73, 0x69, 0x73, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x36, 0x2e, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2e, 0x62,
	0x6f, 0x77, 0x73, 0x65, 0x72, 0x2e, 0x76, 0x31, 0x2e, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73,
	0x6f, 0x72, 0x43, 0x6f, 0x6e, 0x73, 0x75, 0x6d, 0x65, 0x72, 0x41, 0x77, 0x73, 0x4b, 0x69, 0x6e,
	0x65, 0x73, 0x69, 0x73, 0x52, 0x07, 0x6b, 0x69, 0x6e, 0x65, 0x73, 0x69, 0x73, 0x22, 0x1d, 0x0a,
	0x1b, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x43, 0x6f, 0x6e, 0x73, 0x75, 0x6d,
	0x65, 0x72, 0x41, 0x77, 0x73, 0x4b, 0x69, 0x6e, 0x65, 0x73, 0x69, 0x73, 0x22, 0x6b, 0x0a, 0x16,
	0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x43, 0x6f, 0x6e, 0x73, 0x75, 0x6d, 0x65,
	0x72, 0x41, 0x77, 0x73, 0x53, 0x33, 0x12, 0x51, 0x0a, 0x07, 0x62, 0x75, 0x63, 0x6b, 0x65, 0x74,
	0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x37, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73,
	0x2e, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2e, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72,
	0x2e, 0x76, 0x31, 0x2e, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x43, 0x6f, 0x6e,
	0x73, 0x75, 0x6d, 0x65, 0x72, 0x41, 0x77, 0x73, 0x53, 0x33, 0x42, 0x75, 0x63, 0x6b, 0x65, 0x74,
	0x52, 0x07, 0x62, 0x75, 0x63, 0x6b, 0x65, 0x74, 0x73, 0x22, 0x46, 0x0a, 0x1c, 0x50, 0x72, 0x6f,
	0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x43, 0x6f, 0x6e, 0x73, 0x75, 0x6d, 0x65, 0x72, 0x41, 0x77,
	0x73, 0x53, 0x33, 0x42, 0x75, 0x63, 0x6b, 0x65, 0x74, 0x12, 0x12, 0x0a, 0x04, 0x6e, 0x61, 0x6d,
	0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x12, 0x0a,
	0x04, 0x75, 0x72, 0x69, 0x73, 0x18, 0x02, 0x20, 0x03, 0x28, 0x09, 0x52, 0x04, 0x75, 0x72, 0x69,
	0x73, 0x42, 0x35, 0x5a, 0x33, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f,
	0x76, 0x6f, 0x78, 0x65, 0x6c, 0x2d, 0x61, 0x69, 0x2f, 0x76, 0x6f, 0x78, 0x65, 0x6c, 0x2f, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2f, 0x62,
	0x6f, 0x77, 0x73, 0x65, 0x72, 0x2f, 0x76, 0x31, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescOnce sync.Once
	file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescData = file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDesc
)

func file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescGZIP() []byte {
	file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescOnce.Do(func() {
		file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescData = protoimpl.X.CompressGZIP(file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescData)
	})
	return file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDescData
}

var file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes = make([]protoimpl.MessageInfo, 5)
var file_protos_platform_bowser_v1_bowser_config_consumer_proto_goTypes = []interface{}{
	(*ProcessorConsumer)(nil),            // 0: protos.platform.bowser.v1.ProcessorConsumer
	(*ProcessorConsumerAws)(nil),         // 1: protos.platform.bowser.v1.ProcessorConsumerAws
	(*ProcessorConsumerAwsKinesis)(nil),  // 2: protos.platform.bowser.v1.ProcessorConsumerAwsKinesis
	(*ProcessorConsumerAwsS3)(nil),       // 3: protos.platform.bowser.v1.ProcessorConsumerAwsS3
	(*ProcessorConsumerAwsS3Bucket)(nil), // 4: protos.platform.bowser.v1.ProcessorConsumerAwsS3Bucket
}
var file_protos_platform_bowser_v1_bowser_config_consumer_proto_depIdxs = []int32{
	1, // 0: protos.platform.bowser.v1.ProcessorConsumer.aws:type_name -> protos.platform.bowser.v1.ProcessorConsumerAws
	3, // 1: protos.platform.bowser.v1.ProcessorConsumerAws.s3:type_name -> protos.platform.bowser.v1.ProcessorConsumerAwsS3
	2, // 2: protos.platform.bowser.v1.ProcessorConsumerAws.kinesis:type_name -> protos.platform.bowser.v1.ProcessorConsumerAwsKinesis
	4, // 3: protos.platform.bowser.v1.ProcessorConsumerAwsS3.buckets:type_name -> protos.platform.bowser.v1.ProcessorConsumerAwsS3Bucket
	4, // [4:4] is the sub-list for method output_type
	4, // [4:4] is the sub-list for method input_type
	4, // [4:4] is the sub-list for extension type_name
	4, // [4:4] is the sub-list for extension extendee
	0, // [0:4] is the sub-list for field type_name
}

func init() { file_protos_platform_bowser_v1_bowser_config_consumer_proto_init() }
func file_protos_platform_bowser_v1_bowser_config_consumer_proto_init() {
	if File_protos_platform_bowser_v1_bowser_config_consumer_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessorConsumer); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessorConsumerAws); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessorConsumerAwsKinesis); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessorConsumerAwsS3); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessorConsumerAwsS3Bucket); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   5,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_protos_platform_bowser_v1_bowser_config_consumer_proto_goTypes,
		DependencyIndexes: file_protos_platform_bowser_v1_bowser_config_consumer_proto_depIdxs,
		MessageInfos:      file_protos_platform_bowser_v1_bowser_config_consumer_proto_msgTypes,
	}.Build()
	File_protos_platform_bowser_v1_bowser_config_consumer_proto = out.File
	file_protos_platform_bowser_v1_bowser_config_consumer_proto_rawDesc = nil
	file_protos_platform_bowser_v1_bowser_config_consumer_proto_goTypes = nil
	file_protos_platform_bowser_v1_bowser_config_consumer_proto_depIdxs = nil
}