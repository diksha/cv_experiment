// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        v3.20.3
// source: protos/perception/cirrus/v1/repo.proto

package cirruspb

import (
	triton "github.com/voxel-ai/voxel/protos/third_party/triton"
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

type Repository struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Models    []*Model    `protobuf:"bytes,1,rep,name=models,proto3" json:"models,omitempty"`
	Ensembles []*Ensemble `protobuf:"bytes,2,rep,name=ensembles,proto3" json:"ensembles,omitempty"`
}

func (x *Repository) Reset() {
	*x = Repository{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_perception_cirrus_v1_repo_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Repository) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Repository) ProtoMessage() {}

func (x *Repository) ProtoReflect() protoreflect.Message {
	mi := &file_protos_perception_cirrus_v1_repo_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Repository.ProtoReflect.Descriptor instead.
func (*Repository) Descriptor() ([]byte, []int) {
	return file_protos_perception_cirrus_v1_repo_proto_rawDescGZIP(), []int{0}
}

func (x *Repository) GetModels() []*Model {
	if x != nil {
		return x.Models
	}
	return nil
}

func (x *Repository) GetEnsembles() []*Ensemble {
	if x != nil {
		return x.Ensembles
	}
	return nil
}

type Model struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	ArtifactModelPaths      []string            `protobuf:"bytes,1,rep,name=artifact_model_paths,json=artifactModelPaths,proto3" json:"artifact_model_paths,omitempty"`
	Config                  *triton.ModelConfig `protobuf:"bytes,2,opt,name=config,proto3" json:"config,omitempty"`
	DisableWarmupGeneration bool                `protobuf:"varint,4,opt,name=disable_warmup_generation,json=disableWarmupGeneration,proto3" json:"disable_warmup_generation,omitempty"`
}

func (x *Model) Reset() {
	*x = Model{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_perception_cirrus_v1_repo_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Model) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Model) ProtoMessage() {}

func (x *Model) ProtoReflect() protoreflect.Message {
	mi := &file_protos_perception_cirrus_v1_repo_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Model.ProtoReflect.Descriptor instead.
func (*Model) Descriptor() ([]byte, []int) {
	return file_protos_perception_cirrus_v1_repo_proto_rawDescGZIP(), []int{1}
}

func (x *Model) GetArtifactModelPaths() []string {
	if x != nil {
		return x.ArtifactModelPaths
	}
	return nil
}

func (x *Model) GetConfig() *triton.ModelConfig {
	if x != nil {
		return x.Config
	}
	return nil
}

func (x *Model) GetDisableWarmupGeneration() bool {
	if x != nil {
		return x.DisableWarmupGeneration
	}
	return false
}

type Ensemble struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	PrimaryModelName   string              `protobuf:"bytes,1,opt,name=primary_model_name,json=primaryModelName,proto3" json:"primary_model_name,omitempty"`
	ArtifactModelPaths []string            `protobuf:"bytes,2,rep,name=artifact_model_paths,json=artifactModelPaths,proto3" json:"artifact_model_paths,omitempty"`
	Config             *triton.ModelConfig `protobuf:"bytes,3,opt,name=config,proto3" json:"config,omitempty"`
}

func (x *Ensemble) Reset() {
	*x = Ensemble{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_perception_cirrus_v1_repo_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Ensemble) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Ensemble) ProtoMessage() {}

func (x *Ensemble) ProtoReflect() protoreflect.Message {
	mi := &file_protos_perception_cirrus_v1_repo_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Ensemble.ProtoReflect.Descriptor instead.
func (*Ensemble) Descriptor() ([]byte, []int) {
	return file_protos_perception_cirrus_v1_repo_proto_rawDescGZIP(), []int{2}
}

func (x *Ensemble) GetPrimaryModelName() string {
	if x != nil {
		return x.PrimaryModelName
	}
	return ""
}

func (x *Ensemble) GetArtifactModelPaths() []string {
	if x != nil {
		return x.ArtifactModelPaths
	}
	return nil
}

func (x *Ensemble) GetConfig() *triton.ModelConfig {
	if x != nil {
		return x.Config
	}
	return nil
}

var File_protos_perception_cirrus_v1_repo_proto protoreflect.FileDescriptor

var file_protos_perception_cirrus_v1_repo_proto_rawDesc = []byte{
	0x0a, 0x26, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x70, 0x65, 0x72, 0x63, 0x65, 0x70, 0x74,
	0x69, 0x6f, 0x6e, 0x2f, 0x63, 0x69, 0x72, 0x72, 0x75, 0x73, 0x2f, 0x76, 0x31, 0x2f, 0x72, 0x65,
	0x70, 0x6f, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x1b, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73,
	0x2e, 0x70, 0x65, 0x72, 0x63, 0x65, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x2e, 0x63, 0x69, 0x72, 0x72,
	0x75, 0x73, 0x2e, 0x76, 0x31, 0x1a, 0x2c, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x74, 0x68,
	0x69, 0x72, 0x64, 0x5f, 0x70, 0x61, 0x72, 0x74, 0x79, 0x2f, 0x74, 0x72, 0x69, 0x74, 0x6f, 0x6e,
	0x2f, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x5f, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x2e, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x22, 0x8d, 0x01, 0x0a, 0x0a, 0x52, 0x65, 0x70, 0x6f, 0x73, 0x69, 0x74, 0x6f,
	0x72, 0x79, 0x12, 0x3a, 0x0a, 0x06, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x18, 0x01, 0x20, 0x03,
	0x28, 0x0b, 0x32, 0x22, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x70, 0x65, 0x72, 0x63,
	0x65, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x2e, 0x63, 0x69, 0x72, 0x72, 0x75, 0x73, 0x2e, 0x76, 0x31,
	0x2e, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x52, 0x06, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x12, 0x43,
	0x0a, 0x09, 0x65, 0x6e, 0x73, 0x65, 0x6d, 0x62, 0x6c, 0x65, 0x73, 0x18, 0x02, 0x20, 0x03, 0x28,
	0x0b, 0x32, 0x25, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x70, 0x65, 0x72, 0x63, 0x65,
	0x70, 0x74, 0x69, 0x6f, 0x6e, 0x2e, 0x63, 0x69, 0x72, 0x72, 0x75, 0x73, 0x2e, 0x76, 0x31, 0x2e,
	0x45, 0x6e, 0x73, 0x65, 0x6d, 0x62, 0x6c, 0x65, 0x52, 0x09, 0x65, 0x6e, 0x73, 0x65, 0x6d, 0x62,
	0x6c, 0x65, 0x73, 0x22, 0xb5, 0x01, 0x0a, 0x05, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x12, 0x30, 0x0a,
	0x14, 0x61, 0x72, 0x74, 0x69, 0x66, 0x61, 0x63, 0x74, 0x5f, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x5f,
	0x70, 0x61, 0x74, 0x68, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x09, 0x52, 0x12, 0x61, 0x72, 0x74,
	0x69, 0x66, 0x61, 0x63, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x50, 0x61, 0x74, 0x68, 0x73, 0x12,
	0x3e, 0x0a, 0x06, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32,
	0x26, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x74, 0x68, 0x69, 0x72, 0x64, 0x5f, 0x70,
	0x61, 0x72, 0x74, 0x79, 0x2e, 0x74, 0x72, 0x69, 0x74, 0x6f, 0x6e, 0x2e, 0x4d, 0x6f, 0x64, 0x65,
	0x6c, 0x43, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x52, 0x06, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x12,
	0x3a, 0x0a, 0x19, 0x64, 0x69, 0x73, 0x61, 0x62, 0x6c, 0x65, 0x5f, 0x77, 0x61, 0x72, 0x6d, 0x75,
	0x70, 0x5f, 0x67, 0x65, 0x6e, 0x65, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x18, 0x04, 0x20, 0x01,
	0x28, 0x08, 0x52, 0x17, 0x64, 0x69, 0x73, 0x61, 0x62, 0x6c, 0x65, 0x57, 0x61, 0x72, 0x6d, 0x75,
	0x70, 0x47, 0x65, 0x6e, 0x65, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x22, 0xaa, 0x01, 0x0a, 0x08,
	0x45, 0x6e, 0x73, 0x65, 0x6d, 0x62, 0x6c, 0x65, 0x12, 0x2c, 0x0a, 0x12, 0x70, 0x72, 0x69, 0x6d,
	0x61, 0x72, 0x79, 0x5f, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x09, 0x52, 0x10, 0x70, 0x72, 0x69, 0x6d, 0x61, 0x72, 0x79, 0x4d, 0x6f, 0x64,
	0x65, 0x6c, 0x4e, 0x61, 0x6d, 0x65, 0x12, 0x30, 0x0a, 0x14, 0x61, 0x72, 0x74, 0x69, 0x66, 0x61,
	0x63, 0x74, 0x5f, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x5f, 0x70, 0x61, 0x74, 0x68, 0x73, 0x18, 0x02,
	0x20, 0x03, 0x28, 0x09, 0x52, 0x12, 0x61, 0x72, 0x74, 0x69, 0x66, 0x61, 0x63, 0x74, 0x4d, 0x6f,
	0x64, 0x65, 0x6c, 0x50, 0x61, 0x74, 0x68, 0x73, 0x12, 0x3e, 0x0a, 0x06, 0x63, 0x6f, 0x6e, 0x66,
	0x69, 0x67, 0x18, 0x03, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x26, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x73, 0x2e, 0x74, 0x68, 0x69, 0x72, 0x64, 0x5f, 0x70, 0x61, 0x72, 0x74, 0x79, 0x2e, 0x74, 0x72,
	0x69, 0x74, 0x6f, 0x6e, 0x2e, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x43, 0x6f, 0x6e, 0x66, 0x69, 0x67,
	0x52, 0x06, 0x63, 0x6f, 0x6e, 0x66, 0x69, 0x67, 0x42, 0x40, 0x5a, 0x3e, 0x67, 0x69, 0x74, 0x68,
	0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x76, 0x6f, 0x78, 0x65, 0x6c, 0x2d, 0x61, 0x69, 0x2f,
	0x76, 0x6f, 0x78, 0x65, 0x6c, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x70, 0x65, 0x72,
	0x63, 0x65, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x2f, 0x63, 0x69, 0x72, 0x72, 0x75, 0x73, 0x2f, 0x76,
	0x31, 0x3b, 0x63, 0x69, 0x72, 0x72, 0x75, 0x73, 0x70, 0x62, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x33,
}

var (
	file_protos_perception_cirrus_v1_repo_proto_rawDescOnce sync.Once
	file_protos_perception_cirrus_v1_repo_proto_rawDescData = file_protos_perception_cirrus_v1_repo_proto_rawDesc
)

func file_protos_perception_cirrus_v1_repo_proto_rawDescGZIP() []byte {
	file_protos_perception_cirrus_v1_repo_proto_rawDescOnce.Do(func() {
		file_protos_perception_cirrus_v1_repo_proto_rawDescData = protoimpl.X.CompressGZIP(file_protos_perception_cirrus_v1_repo_proto_rawDescData)
	})
	return file_protos_perception_cirrus_v1_repo_proto_rawDescData
}

var file_protos_perception_cirrus_v1_repo_proto_msgTypes = make([]protoimpl.MessageInfo, 3)
var file_protos_perception_cirrus_v1_repo_proto_goTypes = []interface{}{
	(*Repository)(nil),         // 0: protos.perception.cirrus.v1.Repository
	(*Model)(nil),              // 1: protos.perception.cirrus.v1.Model
	(*Ensemble)(nil),           // 2: protos.perception.cirrus.v1.Ensemble
	(*triton.ModelConfig)(nil), // 3: protos.third_party.triton.ModelConfig
}
var file_protos_perception_cirrus_v1_repo_proto_depIdxs = []int32{
	1, // 0: protos.perception.cirrus.v1.Repository.models:type_name -> protos.perception.cirrus.v1.Model
	2, // 1: protos.perception.cirrus.v1.Repository.ensembles:type_name -> protos.perception.cirrus.v1.Ensemble
	3, // 2: protos.perception.cirrus.v1.Model.config:type_name -> protos.third_party.triton.ModelConfig
	3, // 3: protos.perception.cirrus.v1.Ensemble.config:type_name -> protos.third_party.triton.ModelConfig
	4, // [4:4] is the sub-list for method output_type
	4, // [4:4] is the sub-list for method input_type
	4, // [4:4] is the sub-list for extension type_name
	4, // [4:4] is the sub-list for extension extendee
	0, // [0:4] is the sub-list for field type_name
}

func init() { file_protos_perception_cirrus_v1_repo_proto_init() }
func file_protos_perception_cirrus_v1_repo_proto_init() {
	if File_protos_perception_cirrus_v1_repo_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_protos_perception_cirrus_v1_repo_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Repository); i {
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
		file_protos_perception_cirrus_v1_repo_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Model); i {
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
		file_protos_perception_cirrus_v1_repo_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Ensemble); i {
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
			RawDescriptor: file_protos_perception_cirrus_v1_repo_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   3,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_protos_perception_cirrus_v1_repo_proto_goTypes,
		DependencyIndexes: file_protos_perception_cirrus_v1_repo_proto_depIdxs,
		MessageInfos:      file_protos_perception_cirrus_v1_repo_proto_msgTypes,
	}.Build()
	File_protos_perception_cirrus_v1_repo_proto = out.File
	file_protos_perception_cirrus_v1_repo_proto_rawDesc = nil
	file_protos_perception_cirrus_v1_repo_proto_goTypes = nil
	file_protos_perception_cirrus_v1_repo_proto_depIdxs = nil
}
