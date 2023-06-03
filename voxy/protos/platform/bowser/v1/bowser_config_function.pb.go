// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        v3.20.3
// source: protos/platform/bowser/v1/bowser_config_function.proto

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

type ProcessorFunction struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Name      string                      `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Aggregate *ProcessorFunctionAggregate `protobuf:"bytes,2,opt,name=aggregate,proto3" json:"aggregate,omitempty"`
	Reduce    *ProcessorFunctionReduce    `protobuf:"bytes,3,opt,name=reduce,proto3" json:"reduce,omitempty"`
}

func (x *ProcessorFunction) Reset() {
	*x = ProcessorFunction{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_platform_bowser_v1_bowser_config_function_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessorFunction) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessorFunction) ProtoMessage() {}

func (x *ProcessorFunction) ProtoReflect() protoreflect.Message {
	mi := &file_protos_platform_bowser_v1_bowser_config_function_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessorFunction.ProtoReflect.Descriptor instead.
func (*ProcessorFunction) Descriptor() ([]byte, []int) {
	return file_protos_platform_bowser_v1_bowser_config_function_proto_rawDescGZIP(), []int{0}
}

func (x *ProcessorFunction) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *ProcessorFunction) GetAggregate() *ProcessorFunctionAggregate {
	if x != nil {
		return x.Aggregate
	}
	return nil
}

func (x *ProcessorFunction) GetReduce() *ProcessorFunctionReduce {
	if x != nil {
		return x.Reduce
	}
	return nil
}

type ProcessorFunctionAggregate struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *ProcessorFunctionAggregate) Reset() {
	*x = ProcessorFunctionAggregate{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_platform_bowser_v1_bowser_config_function_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessorFunctionAggregate) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessorFunctionAggregate) ProtoMessage() {}

func (x *ProcessorFunctionAggregate) ProtoReflect() protoreflect.Message {
	mi := &file_protos_platform_bowser_v1_bowser_config_function_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessorFunctionAggregate.ProtoReflect.Descriptor instead.
func (*ProcessorFunctionAggregate) Descriptor() ([]byte, []int) {
	return file_protos_platform_bowser_v1_bowser_config_function_proto_rawDescGZIP(), []int{1}
}

type ProcessorFunctionReduce struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *ProcessorFunctionReduce) Reset() {
	*x = ProcessorFunctionReduce{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_platform_bowser_v1_bowser_config_function_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessorFunctionReduce) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessorFunctionReduce) ProtoMessage() {}

func (x *ProcessorFunctionReduce) ProtoReflect() protoreflect.Message {
	mi := &file_protos_platform_bowser_v1_bowser_config_function_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessorFunctionReduce.ProtoReflect.Descriptor instead.
func (*ProcessorFunctionReduce) Descriptor() ([]byte, []int) {
	return file_protos_platform_bowser_v1_bowser_config_function_proto_rawDescGZIP(), []int{2}
}

var File_protos_platform_bowser_v1_bowser_config_function_proto protoreflect.FileDescriptor

var file_protos_platform_bowser_v1_bowser_config_function_proto_rawDesc = []byte{
	0x0a, 0x36, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72,
	0x6d, 0x2f, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72, 0x2f, 0x76, 0x31, 0x2f, 0x62, 0x6f, 0x77, 0x73,
	0x65, 0x72, 0x5f, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x5f, 0x66, 0x75, 0x6e, 0x63, 0x74, 0x69,
	0x6f, 0x6e, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x19, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73,
	0x2e, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2e, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72,
	0x2e, 0x76, 0x31, 0x22, 0xc8, 0x01, 0x0a, 0x11, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f,
	0x72, 0x46, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x12, 0x12, 0x0a, 0x04, 0x6e, 0x61, 0x6d,
	0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x53, 0x0a,
	0x09, 0x61, 0x67, 0x67, 0x72, 0x65, 0x67, 0x61, 0x74, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b,
	0x32, 0x35, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f,
	0x72, 0x6d, 0x2e, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72, 0x2e, 0x76, 0x31, 0x2e, 0x50, 0x72, 0x6f,
	0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x46, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x41, 0x67,
	0x67, 0x72, 0x65, 0x67, 0x61, 0x74, 0x65, 0x52, 0x09, 0x61, 0x67, 0x67, 0x72, 0x65, 0x67, 0x61,
	0x74, 0x65, 0x12, 0x4a, 0x0a, 0x06, 0x72, 0x65, 0x64, 0x75, 0x63, 0x65, 0x18, 0x03, 0x20, 0x01,
	0x28, 0x0b, 0x32, 0x32, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x70, 0x6c, 0x61, 0x74,
	0x66, 0x6f, 0x72, 0x6d, 0x2e, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72, 0x2e, 0x76, 0x31, 0x2e, 0x50,
	0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x46, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e,
	0x52, 0x65, 0x64, 0x75, 0x63, 0x65, 0x52, 0x06, 0x72, 0x65, 0x64, 0x75, 0x63, 0x65, 0x22, 0x1c,
	0x0a, 0x1a, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x46, 0x75, 0x6e, 0x63, 0x74,
	0x69, 0x6f, 0x6e, 0x41, 0x67, 0x67, 0x72, 0x65, 0x67, 0x61, 0x74, 0x65, 0x22, 0x19, 0x0a, 0x17,
	0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x46, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f,
	0x6e, 0x52, 0x65, 0x64, 0x75, 0x63, 0x65, 0x42, 0x35, 0x5a, 0x33, 0x67, 0x69, 0x74, 0x68, 0x75,
	0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x76, 0x6f, 0x78, 0x65, 0x6c, 0x2d, 0x61, 0x69, 0x2f, 0x76,
	0x6f, 0x78, 0x65, 0x6c, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x70, 0x6c, 0x61, 0x74,
	0x66, 0x6f, 0x72, 0x6d, 0x2f, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72, 0x2f, 0x76, 0x31, 0x62, 0x06,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_protos_platform_bowser_v1_bowser_config_function_proto_rawDescOnce sync.Once
	file_protos_platform_bowser_v1_bowser_config_function_proto_rawDescData = file_protos_platform_bowser_v1_bowser_config_function_proto_rawDesc
)

func file_protos_platform_bowser_v1_bowser_config_function_proto_rawDescGZIP() []byte {
	file_protos_platform_bowser_v1_bowser_config_function_proto_rawDescOnce.Do(func() {
		file_protos_platform_bowser_v1_bowser_config_function_proto_rawDescData = protoimpl.X.CompressGZIP(file_protos_platform_bowser_v1_bowser_config_function_proto_rawDescData)
	})
	return file_protos_platform_bowser_v1_bowser_config_function_proto_rawDescData
}

var file_protos_platform_bowser_v1_bowser_config_function_proto_msgTypes = make([]protoimpl.MessageInfo, 3)
var file_protos_platform_bowser_v1_bowser_config_function_proto_goTypes = []interface{}{
	(*ProcessorFunction)(nil),          // 0: protos.platform.bowser.v1.ProcessorFunction
	(*ProcessorFunctionAggregate)(nil), // 1: protos.platform.bowser.v1.ProcessorFunctionAggregate
	(*ProcessorFunctionReduce)(nil),    // 2: protos.platform.bowser.v1.ProcessorFunctionReduce
}
var file_protos_platform_bowser_v1_bowser_config_function_proto_depIdxs = []int32{
	1, // 0: protos.platform.bowser.v1.ProcessorFunction.aggregate:type_name -> protos.platform.bowser.v1.ProcessorFunctionAggregate
	2, // 1: protos.platform.bowser.v1.ProcessorFunction.reduce:type_name -> protos.platform.bowser.v1.ProcessorFunctionReduce
	2, // [2:2] is the sub-list for method output_type
	2, // [2:2] is the sub-list for method input_type
	2, // [2:2] is the sub-list for extension type_name
	2, // [2:2] is the sub-list for extension extendee
	0, // [0:2] is the sub-list for field type_name
}

func init() { file_protos_platform_bowser_v1_bowser_config_function_proto_init() }
func file_protos_platform_bowser_v1_bowser_config_function_proto_init() {
	if File_protos_platform_bowser_v1_bowser_config_function_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_protos_platform_bowser_v1_bowser_config_function_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessorFunction); i {
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
		file_protos_platform_bowser_v1_bowser_config_function_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessorFunctionAggregate); i {
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
		file_protos_platform_bowser_v1_bowser_config_function_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessorFunctionReduce); i {
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
			RawDescriptor: file_protos_platform_bowser_v1_bowser_config_function_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   3,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_protos_platform_bowser_v1_bowser_config_function_proto_goTypes,
		DependencyIndexes: file_protos_platform_bowser_v1_bowser_config_function_proto_depIdxs,
		MessageInfos:      file_protos_platform_bowser_v1_bowser_config_function_proto_msgTypes,
	}.Build()
	File_protos_platform_bowser_v1_bowser_config_function_proto = out.File
	file_protos_platform_bowser_v1_bowser_config_function_proto_rawDesc = nil
	file_protos_platform_bowser_v1_bowser_config_function_proto_goTypes = nil
	file_protos_platform_bowser_v1_bowser_config_function_proto_depIdxs = nil
}