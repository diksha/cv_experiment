// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        v3.20.3
// source: protos/perception/cirrus/v1/cirrus_deployment.proto

package cirruspb

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

type DeploymentConfig struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Cameras []*Camera `protobuf:"bytes,1,rep,name=cameras,proto3" json:"cameras,omitempty"`
}

func (x *DeploymentConfig) Reset() {
	*x = DeploymentConfig{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_perception_cirrus_v1_cirrus_deployment_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DeploymentConfig) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeploymentConfig) ProtoMessage() {}

func (x *DeploymentConfig) ProtoReflect() protoreflect.Message {
	mi := &file_protos_perception_cirrus_v1_cirrus_deployment_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeploymentConfig.ProtoReflect.Descriptor instead.
func (*DeploymentConfig) Descriptor() ([]byte, []int) {
	return file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDescGZIP(), []int{0}
}

func (x *DeploymentConfig) GetCameras() []*Camera {
	if x != nil {
		return x.Cameras
	}
	return nil
}

type Camera struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	CameraConfigPath string `protobuf:"bytes,1,opt,name=camera_config_path,json=cameraConfigPath,proto3" json:"camera_config_path,omitempty"`
}

func (x *Camera) Reset() {
	*x = Camera{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_perception_cirrus_v1_cirrus_deployment_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Camera) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Camera) ProtoMessage() {}

func (x *Camera) ProtoReflect() protoreflect.Message {
	mi := &file_protos_perception_cirrus_v1_cirrus_deployment_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Camera.ProtoReflect.Descriptor instead.
func (*Camera) Descriptor() ([]byte, []int) {
	return file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDescGZIP(), []int{1}
}

func (x *Camera) GetCameraConfigPath() string {
	if x != nil {
		return x.CameraConfigPath
	}
	return ""
}

var File_protos_perception_cirrus_v1_cirrus_deployment_proto protoreflect.FileDescriptor

var file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDesc = []byte{
	0x0a, 0x33, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x70, 0x65, 0x72, 0x63, 0x65, 0x70, 0x74,
	0x69, 0x6f, 0x6e, 0x2f, 0x63, 0x69, 0x72, 0x72, 0x75, 0x73, 0x2f, 0x76, 0x31, 0x2f, 0x63, 0x69,
	0x72, 0x72, 0x75, 0x73, 0x5f, 0x64, 0x65, 0x70, 0x6c, 0x6f, 0x79, 0x6d, 0x65, 0x6e, 0x74, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x1b, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x70, 0x65,
	0x72, 0x63, 0x65, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x2e, 0x63, 0x69, 0x72, 0x72, 0x75, 0x73, 0x2e,
	0x76, 0x31, 0x22, 0x51, 0x0a, 0x10, 0x44, 0x65, 0x70, 0x6c, 0x6f, 0x79, 0x6d, 0x65, 0x6e, 0x74,
	0x43, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x12, 0x3d, 0x0a, 0x07, 0x63, 0x61, 0x6d, 0x65, 0x72, 0x61,
	0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x23, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73,
	0x2e, 0x70, 0x65, 0x72, 0x63, 0x65, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x2e, 0x63, 0x69, 0x72, 0x72,
	0x75, 0x73, 0x2e, 0x76, 0x31, 0x2e, 0x43, 0x61, 0x6d, 0x65, 0x72, 0x61, 0x52, 0x07, 0x63, 0x61,
	0x6d, 0x65, 0x72, 0x61, 0x73, 0x22, 0x36, 0x0a, 0x06, 0x43, 0x61, 0x6d, 0x65, 0x72, 0x61, 0x12,
	0x2c, 0x0a, 0x12, 0x63, 0x61, 0x6d, 0x65, 0x72, 0x61, 0x5f, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67,
	0x5f, 0x70, 0x61, 0x74, 0x68, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x10, 0x63, 0x61, 0x6d,
	0x65, 0x72, 0x61, 0x43, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x50, 0x61, 0x74, 0x68, 0x42, 0x40, 0x5a,
	0x3e, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x76, 0x6f, 0x78, 0x65,
	0x6c, 0x2d, 0x61, 0x69, 0x2f, 0x76, 0x6f, 0x78, 0x65, 0x6c, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x73, 0x2f, 0x70, 0x65, 0x72, 0x63, 0x65, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x2f, 0x63, 0x69, 0x72,
	0x72, 0x75, 0x73, 0x2f, 0x76, 0x31, 0x3b, 0x63, 0x69, 0x72, 0x72, 0x75, 0x73, 0x70, 0x62, 0x62,
	0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDescOnce sync.Once
	file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDescData = file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDesc
)

func file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDescGZIP() []byte {
	file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDescOnce.Do(func() {
		file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDescData = protoimpl.X.CompressGZIP(file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDescData)
	})
	return file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDescData
}

var file_protos_perception_cirrus_v1_cirrus_deployment_proto_msgTypes = make([]protoimpl.MessageInfo, 2)
var file_protos_perception_cirrus_v1_cirrus_deployment_proto_goTypes = []interface{}{
	(*DeploymentConfig)(nil), // 0: protos.perception.cirrus.v1.DeploymentConfig
	(*Camera)(nil),           // 1: protos.perception.cirrus.v1.Camera
}
var file_protos_perception_cirrus_v1_cirrus_deployment_proto_depIdxs = []int32{
	1, // 0: protos.perception.cirrus.v1.DeploymentConfig.cameras:type_name -> protos.perception.cirrus.v1.Camera
	1, // [1:1] is the sub-list for method output_type
	1, // [1:1] is the sub-list for method input_type
	1, // [1:1] is the sub-list for extension type_name
	1, // [1:1] is the sub-list for extension extendee
	0, // [0:1] is the sub-list for field type_name
}

func init() { file_protos_perception_cirrus_v1_cirrus_deployment_proto_init() }
func file_protos_perception_cirrus_v1_cirrus_deployment_proto_init() {
	if File_protos_perception_cirrus_v1_cirrus_deployment_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_protos_perception_cirrus_v1_cirrus_deployment_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*DeploymentConfig); i {
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
		file_protos_perception_cirrus_v1_cirrus_deployment_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Camera); i {
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
			RawDescriptor: file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   2,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_protos_perception_cirrus_v1_cirrus_deployment_proto_goTypes,
		DependencyIndexes: file_protos_perception_cirrus_v1_cirrus_deployment_proto_depIdxs,
		MessageInfos:      file_protos_perception_cirrus_v1_cirrus_deployment_proto_msgTypes,
	}.Build()
	File_protos_perception_cirrus_v1_cirrus_deployment_proto = out.File
	file_protos_perception_cirrus_v1_cirrus_deployment_proto_rawDesc = nil
	file_protos_perception_cirrus_v1_cirrus_deployment_proto_goTypes = nil
	file_protos_perception_cirrus_v1_cirrus_deployment_proto_depIdxs = nil
}
