// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        v3.20.3
// source: protos/platform/bowser/v1/bowser_config_keys.proto

package v1

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	descriptorpb "google.golang.org/protobuf/types/descriptorpb"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type Unit int32

const (
	Unit_UNIT_MINUTE_UNSPECIFIED Unit = 0
	Unit_UNIT_FIVE_MINUTE        Unit = 1
	Unit_UNIT_HALF_HOUR          Unit = 2
	Unit_UNIT_HOURS              Unit = 3
	Unit_UNIT_DAY                Unit = 4
	Unit_UNIT_WEEK               Unit = 5
	Unit_UNIT_MONTH              Unit = 6
)

// Enum value maps for Unit.
var (
	Unit_name = map[int32]string{
		0: "UNIT_MINUTE_UNSPECIFIED",
		1: "UNIT_FIVE_MINUTE",
		2: "UNIT_HALF_HOUR",
		3: "UNIT_HOURS",
		4: "UNIT_DAY",
		5: "UNIT_WEEK",
		6: "UNIT_MONTH",
	}
	Unit_value = map[string]int32{
		"UNIT_MINUTE_UNSPECIFIED": 0,
		"UNIT_FIVE_MINUTE":        1,
		"UNIT_HALF_HOUR":          2,
		"UNIT_HOURS":              3,
		"UNIT_DAY":                4,
		"UNIT_WEEK":               5,
		"UNIT_MONTH":              6,
	}
)

func (x Unit) Enum() *Unit {
	p := new(Unit)
	*p = x
	return p
}

func (x Unit) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (Unit) Descriptor() protoreflect.EnumDescriptor {
	return file_protos_platform_bowser_v1_bowser_config_keys_proto_enumTypes[0].Descriptor()
}

func (Unit) Type() protoreflect.EnumType {
	return &file_protos_platform_bowser_v1_bowser_config_keys_proto_enumTypes[0]
}

func (x Unit) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use Unit.Descriptor instead.
func (Unit) EnumDescriptor() ([]byte, []int) {
	return file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDescGZIP(), []int{0}
}

type ProcessorKeys struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Name      []string                             `protobuf:"bytes,1,rep,name=name,proto3" json:"name,omitempty"`
	Fields    []string                             `protobuf:"bytes,2,rep,name=fields,proto3" json:"fields,omitempty"`
	Timestamp *ProcessorFunctionKeyTimestampConfig `protobuf:"bytes,3,opt,name=timestamp,proto3" json:"timestamp,omitempty"`
}

func (x *ProcessorKeys) Reset() {
	*x = ProcessorKeys{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_platform_bowser_v1_bowser_config_keys_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessorKeys) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessorKeys) ProtoMessage() {}

func (x *ProcessorKeys) ProtoReflect() protoreflect.Message {
	mi := &file_protos_platform_bowser_v1_bowser_config_keys_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessorKeys.ProtoReflect.Descriptor instead.
func (*ProcessorKeys) Descriptor() ([]byte, []int) {
	return file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDescGZIP(), []int{0}
}

func (x *ProcessorKeys) GetName() []string {
	if x != nil {
		return x.Name
	}
	return nil
}

func (x *ProcessorKeys) GetFields() []string {
	if x != nil {
		return x.Fields
	}
	return nil
}

func (x *ProcessorKeys) GetTimestamp() *ProcessorFunctionKeyTimestampConfig {
	if x != nil {
		return x.Timestamp
	}
	return nil
}

type ProcessorFunctionKeyTimestampConfig struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Field string `protobuf:"bytes,1,opt,name=field,proto3" json:"field,omitempty"`
	By    Unit   `protobuf:"varint,2,opt,name=by,proto3,enum=protos.platform.bowser.v1.Unit" json:"by,omitempty"`
}

func (x *ProcessorFunctionKeyTimestampConfig) Reset() {
	*x = ProcessorFunctionKeyTimestampConfig{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_platform_bowser_v1_bowser_config_keys_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessorFunctionKeyTimestampConfig) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessorFunctionKeyTimestampConfig) ProtoMessage() {}

func (x *ProcessorFunctionKeyTimestampConfig) ProtoReflect() protoreflect.Message {
	mi := &file_protos_platform_bowser_v1_bowser_config_keys_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessorFunctionKeyTimestampConfig.ProtoReflect.Descriptor instead.
func (*ProcessorFunctionKeyTimestampConfig) Descriptor() ([]byte, []int) {
	return file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDescGZIP(), []int{1}
}

func (x *ProcessorFunctionKeyTimestampConfig) GetField() string {
	if x != nil {
		return x.Field
	}
	return ""
}

func (x *ProcessorFunctionKeyTimestampConfig) GetBy() Unit {
	if x != nil {
		return x.By
	}
	return Unit_UNIT_MINUTE_UNSPECIFIED
}

var file_protos_platform_bowser_v1_bowser_config_keys_proto_extTypes = []protoimpl.ExtensionInfo{
	{
		ExtendedType:  (*descriptorpb.EnumValueOptions)(nil),
		ExtensionType: (*int32)(nil),
		Field:         51234,
		Name:          "protos.platform.bowser.v1.seconds",
		Tag:           "varint,51234,opt,name=seconds",
		Filename:      "protos/platform/bowser/v1/bowser_config_keys.proto",
	},
}

// Extension fields to descriptorpb.EnumValueOptions.
var (
	// optional int32 seconds = 51234;
	E_Seconds = &file_protos_platform_bowser_v1_bowser_config_keys_proto_extTypes[0]
)

var File_protos_platform_bowser_v1_bowser_config_keys_proto protoreflect.FileDescriptor

var file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDesc = []byte{
	0x0a, 0x32, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72,
	0x6d, 0x2f, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72, 0x2f, 0x76, 0x31, 0x2f, 0x62, 0x6f, 0x77, 0x73,
	0x65, 0x72, 0x5f, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x5f, 0x6b, 0x65, 0x79, 0x73, 0x2e, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x12, 0x19, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x70, 0x6c, 0x61,
	0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2e, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72, 0x2e, 0x76, 0x31, 0x1a,
	0x20, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66,
	0x2f, 0x64, 0x65, 0x73, 0x63, 0x72, 0x69, 0x70, 0x74, 0x6f, 0x72, 0x2e, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x22, 0x99, 0x01, 0x0a, 0x0d, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x4b,
	0x65, 0x79, 0x73, 0x12, 0x12, 0x0a, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x03, 0x28,
	0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x16, 0x0a, 0x06, 0x66, 0x69, 0x65, 0x6c, 0x64,
	0x73, 0x18, 0x02, 0x20, 0x03, 0x28, 0x09, 0x52, 0x06, 0x66, 0x69, 0x65, 0x6c, 0x64, 0x73, 0x12,
	0x5c, 0x0a, 0x09, 0x74, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x18, 0x03, 0x20, 0x01,
	0x28, 0x0b, 0x32, 0x3e, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x70, 0x6c, 0x61, 0x74,
	0x66, 0x6f, 0x72, 0x6d, 0x2e, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72, 0x2e, 0x76, 0x31, 0x2e, 0x50,
	0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x46, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e,
	0x4b, 0x65, 0x79, 0x54, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x43, 0x6f, 0x6e, 0x66,
	0x69, 0x67, 0x52, 0x09, 0x74, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x22, 0x6c, 0x0a,
	0x23, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x6f, 0x72, 0x46, 0x75, 0x6e, 0x63, 0x74, 0x69,
	0x6f, 0x6e, 0x4b, 0x65, 0x79, 0x54, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x43, 0x6f,
	0x6e, 0x66, 0x69, 0x67, 0x12, 0x14, 0x0a, 0x05, 0x66, 0x69, 0x65, 0x6c, 0x64, 0x18, 0x01, 0x20,
	0x01, 0x28, 0x09, 0x52, 0x05, 0x66, 0x69, 0x65, 0x6c, 0x64, 0x12, 0x2f, 0x0a, 0x02, 0x62, 0x79,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x0e, 0x32, 0x1f, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e,
	0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2e, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72, 0x2e,
	0x76, 0x31, 0x2e, 0x55, 0x6e, 0x69, 0x74, 0x52, 0x02, 0x62, 0x79, 0x2a, 0xad, 0x01, 0x0a, 0x04,
	0x55, 0x6e, 0x69, 0x74, 0x12, 0x21, 0x0a, 0x17, 0x55, 0x4e, 0x49, 0x54, 0x5f, 0x4d, 0x49, 0x4e,
	0x55, 0x54, 0x45, 0x5f, 0x55, 0x4e, 0x53, 0x50, 0x45, 0x43, 0x49, 0x46, 0x49, 0x45, 0x44, 0x10,
	0x00, 0x1a, 0x04, 0x90, 0x82, 0x19, 0x3c, 0x12, 0x1b, 0x0a, 0x10, 0x55, 0x4e, 0x49, 0x54, 0x5f,
	0x46, 0x49, 0x56, 0x45, 0x5f, 0x4d, 0x49, 0x4e, 0x55, 0x54, 0x45, 0x10, 0x01, 0x1a, 0x05, 0x90,
	0x82, 0x19, 0xac, 0x02, 0x12, 0x19, 0x0a, 0x0e, 0x55, 0x4e, 0x49, 0x54, 0x5f, 0x48, 0x41, 0x4c,
	0x46, 0x5f, 0x48, 0x4f, 0x55, 0x52, 0x10, 0x02, 0x1a, 0x05, 0x90, 0x82, 0x19, 0x88, 0x0e, 0x12,
	0x15, 0x0a, 0x0a, 0x55, 0x4e, 0x49, 0x54, 0x5f, 0x48, 0x4f, 0x55, 0x52, 0x53, 0x10, 0x03, 0x1a,
	0x05, 0x90, 0x82, 0x19, 0x90, 0x1c, 0x12, 0x14, 0x0a, 0x08, 0x55, 0x4e, 0x49, 0x54, 0x5f, 0x44,
	0x41, 0x59, 0x10, 0x04, 0x1a, 0x06, 0x90, 0x82, 0x19, 0x80, 0xa3, 0x05, 0x12, 0x0d, 0x0a, 0x09,
	0x55, 0x4e, 0x49, 0x54, 0x5f, 0x57, 0x45, 0x45, 0x4b, 0x10, 0x05, 0x12, 0x0e, 0x0a, 0x0a, 0x55,
	0x4e, 0x49, 0x54, 0x5f, 0x4d, 0x4f, 0x4e, 0x54, 0x48, 0x10, 0x06, 0x3a, 0x3d, 0x0a, 0x07, 0x73,
	0x65, 0x63, 0x6f, 0x6e, 0x64, 0x73, 0x12, 0x21, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2e, 0x45, 0x6e, 0x75, 0x6d, 0x56, 0x61, 0x6c,
	0x75, 0x65, 0x4f, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x18, 0xa2, 0x90, 0x03, 0x20, 0x01, 0x28,
	0x05, 0x52, 0x07, 0x73, 0x65, 0x63, 0x6f, 0x6e, 0x64, 0x73, 0x42, 0x35, 0x5a, 0x33, 0x67, 0x69,
	0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x76, 0x6f, 0x78, 0x65, 0x6c, 0x2d, 0x61,
	0x69, 0x2f, 0x76, 0x6f, 0x78, 0x65, 0x6c, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x70,
	0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2f, 0x62, 0x6f, 0x77, 0x73, 0x65, 0x72, 0x2f, 0x76,
	0x31, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDescOnce sync.Once
	file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDescData = file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDesc
)

func file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDescGZIP() []byte {
	file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDescOnce.Do(func() {
		file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDescData = protoimpl.X.CompressGZIP(file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDescData)
	})
	return file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDescData
}

var file_protos_platform_bowser_v1_bowser_config_keys_proto_enumTypes = make([]protoimpl.EnumInfo, 1)
var file_protos_platform_bowser_v1_bowser_config_keys_proto_msgTypes = make([]protoimpl.MessageInfo, 2)
var file_protos_platform_bowser_v1_bowser_config_keys_proto_goTypes = []interface{}{
	(Unit)(0),             // 0: protos.platform.bowser.v1.Unit
	(*ProcessorKeys)(nil), // 1: protos.platform.bowser.v1.ProcessorKeys
	(*ProcessorFunctionKeyTimestampConfig)(nil), // 2: protos.platform.bowser.v1.ProcessorFunctionKeyTimestampConfig
	(*descriptorpb.EnumValueOptions)(nil),       // 3: google.protobuf.EnumValueOptions
}
var file_protos_platform_bowser_v1_bowser_config_keys_proto_depIdxs = []int32{
	2, // 0: protos.platform.bowser.v1.ProcessorKeys.timestamp:type_name -> protos.platform.bowser.v1.ProcessorFunctionKeyTimestampConfig
	0, // 1: protos.platform.bowser.v1.ProcessorFunctionKeyTimestampConfig.by:type_name -> protos.platform.bowser.v1.Unit
	3, // 2: protos.platform.bowser.v1.seconds:extendee -> google.protobuf.EnumValueOptions
	3, // [3:3] is the sub-list for method output_type
	3, // [3:3] is the sub-list for method input_type
	3, // [3:3] is the sub-list for extension type_name
	2, // [2:3] is the sub-list for extension extendee
	0, // [0:2] is the sub-list for field type_name
}

func init() { file_protos_platform_bowser_v1_bowser_config_keys_proto_init() }
func file_protos_platform_bowser_v1_bowser_config_keys_proto_init() {
	if File_protos_platform_bowser_v1_bowser_config_keys_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_protos_platform_bowser_v1_bowser_config_keys_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessorKeys); i {
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
		file_protos_platform_bowser_v1_bowser_config_keys_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessorFunctionKeyTimestampConfig); i {
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
			RawDescriptor: file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDesc,
			NumEnums:      1,
			NumMessages:   2,
			NumExtensions: 1,
			NumServices:   0,
		},
		GoTypes:           file_protos_platform_bowser_v1_bowser_config_keys_proto_goTypes,
		DependencyIndexes: file_protos_platform_bowser_v1_bowser_config_keys_proto_depIdxs,
		EnumInfos:         file_protos_platform_bowser_v1_bowser_config_keys_proto_enumTypes,
		MessageInfos:      file_protos_platform_bowser_v1_bowser_config_keys_proto_msgTypes,
		ExtensionInfos:    file_protos_platform_bowser_v1_bowser_config_keys_proto_extTypes,
	}.Build()
	File_protos_platform_bowser_v1_bowser_config_keys_proto = out.File
	file_protos_platform_bowser_v1_bowser_config_keys_proto_rawDesc = nil
	file_protos_platform_bowser_v1_bowser_config_keys_proto_goTypes = nil
	file_protos_platform_bowser_v1_bowser_config_keys_proto_depIdxs = nil
}