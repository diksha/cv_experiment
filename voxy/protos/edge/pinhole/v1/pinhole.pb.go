// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        v3.20.3
// source: protos/edge/pinhole/v1/pinhole.proto

package pinholepb

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	timestamppb "google.golang.org/protobuf/types/known/timestamppb"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type SignRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Csr string `protobuf:"bytes,1,opt,name=csr,proto3" json:"csr,omitempty"`
}

func (x *SignRequest) Reset() {
	*x = SignRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_edge_pinhole_v1_pinhole_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SignRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SignRequest) ProtoMessage() {}

func (x *SignRequest) ProtoReflect() protoreflect.Message {
	mi := &file_protos_edge_pinhole_v1_pinhole_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SignRequest.ProtoReflect.Descriptor instead.
func (*SignRequest) Descriptor() ([]byte, []int) {
	return file_protos_edge_pinhole_v1_pinhole_proto_rawDescGZIP(), []int{0}
}

func (x *SignRequest) GetCsr() string {
	if x != nil {
		return x.Csr
	}
	return ""
}

type SignResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Cert       string                 `protobuf:"bytes,1,opt,name=cert,proto3" json:"cert,omitempty"`
	RootCa     string                 `protobuf:"bytes,2,opt,name=root_ca,json=rootCa,proto3" json:"root_ca,omitempty"`
	Expiration *timestamppb.Timestamp `protobuf:"bytes,3,opt,name=expiration,proto3" json:"expiration,omitempty"`
}

func (x *SignResponse) Reset() {
	*x = SignResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_protos_edge_pinhole_v1_pinhole_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SignResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SignResponse) ProtoMessage() {}

func (x *SignResponse) ProtoReflect() protoreflect.Message {
	mi := &file_protos_edge_pinhole_v1_pinhole_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SignResponse.ProtoReflect.Descriptor instead.
func (*SignResponse) Descriptor() ([]byte, []int) {
	return file_protos_edge_pinhole_v1_pinhole_proto_rawDescGZIP(), []int{1}
}

func (x *SignResponse) GetCert() string {
	if x != nil {
		return x.Cert
	}
	return ""
}

func (x *SignResponse) GetRootCa() string {
	if x != nil {
		return x.RootCa
	}
	return ""
}

func (x *SignResponse) GetExpiration() *timestamppb.Timestamp {
	if x != nil {
		return x.Expiration
	}
	return nil
}

var File_protos_edge_pinhole_v1_pinhole_proto protoreflect.FileDescriptor

var file_protos_edge_pinhole_v1_pinhole_proto_rawDesc = []byte{
	0x0a, 0x24, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2f, 0x65, 0x64, 0x67, 0x65, 0x2f, 0x70, 0x69,
	0x6e, 0x68, 0x6f, 0x6c, 0x65, 0x2f, 0x76, 0x31, 0x2f, 0x70, 0x69, 0x6e, 0x68, 0x6f, 0x6c, 0x65,
	0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x16, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x65,
	0x64, 0x67, 0x65, 0x2e, 0x70, 0x69, 0x6e, 0x68, 0x6f, 0x6c, 0x65, 0x2e, 0x76, 0x31, 0x1a, 0x1f,
	0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2f,
	0x74, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22,
	0x1f, 0x0a, 0x0b, 0x53, 0x69, 0x67, 0x6e, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x10,
	0x0a, 0x03, 0x63, 0x73, 0x72, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03, 0x63, 0x73, 0x72,
	0x22, 0x77, 0x0a, 0x0c, 0x53, 0x69, 0x67, 0x6e, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65,
	0x12, 0x12, 0x0a, 0x04, 0x63, 0x65, 0x72, 0x74, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04,
	0x63, 0x65, 0x72, 0x74, 0x12, 0x17, 0x0a, 0x07, 0x72, 0x6f, 0x6f, 0x74, 0x5f, 0x63, 0x61, 0x18,
	0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x72, 0x6f, 0x6f, 0x74, 0x43, 0x61, 0x12, 0x3a, 0x0a,
	0x0a, 0x65, 0x78, 0x70, 0x69, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x18, 0x03, 0x20, 0x01, 0x28,
	0x0b, 0x32, 0x1a, 0x2e, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x62, 0x75, 0x66, 0x2e, 0x54, 0x69, 0x6d, 0x65, 0x73, 0x74, 0x61, 0x6d, 0x70, 0x52, 0x0a, 0x65,
	0x78, 0x70, 0x69, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x32, 0x63, 0x0a, 0x0e, 0x50, 0x69, 0x6e,
	0x68, 0x6f, 0x6c, 0x65, 0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x12, 0x51, 0x0a, 0x04, 0x53,
	0x69, 0x67, 0x6e, 0x12, 0x23, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x2e, 0x65, 0x64, 0x67,
	0x65, 0x2e, 0x70, 0x69, 0x6e, 0x68, 0x6f, 0x6c, 0x65, 0x2e, 0x76, 0x31, 0x2e, 0x53, 0x69, 0x67,
	0x6e, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x24, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x73, 0x2e, 0x65, 0x64, 0x67, 0x65, 0x2e, 0x70, 0x69, 0x6e, 0x68, 0x6f, 0x6c, 0x65, 0x2e, 0x76,
	0x31, 0x2e, 0x53, 0x69, 0x67, 0x6e, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x42, 0x40,
	0x5a, 0x3e, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x76, 0x6f, 0x78,
	0x65, 0x6c, 0x2d, 0x61, 0x69, 0x2f, 0x76, 0x6f, 0x78, 0x65, 0x6c, 0x2f, 0x70, 0x72, 0x6f, 0x74,
	0x6f, 0x73, 0x2f, 0x70, 0x6c, 0x61, 0x74, 0x66, 0x6f, 0x72, 0x6d, 0x2f, 0x70, 0x69, 0x6e, 0x68,
	0x6f, 0x6c, 0x65, 0x2f, 0x76, 0x31, 0x3b, 0x70, 0x69, 0x6e, 0x68, 0x6f, 0x6c, 0x65, 0x70, 0x62,
	0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_protos_edge_pinhole_v1_pinhole_proto_rawDescOnce sync.Once
	file_protos_edge_pinhole_v1_pinhole_proto_rawDescData = file_protos_edge_pinhole_v1_pinhole_proto_rawDesc
)

func file_protos_edge_pinhole_v1_pinhole_proto_rawDescGZIP() []byte {
	file_protos_edge_pinhole_v1_pinhole_proto_rawDescOnce.Do(func() {
		file_protos_edge_pinhole_v1_pinhole_proto_rawDescData = protoimpl.X.CompressGZIP(file_protos_edge_pinhole_v1_pinhole_proto_rawDescData)
	})
	return file_protos_edge_pinhole_v1_pinhole_proto_rawDescData
}

var file_protos_edge_pinhole_v1_pinhole_proto_msgTypes = make([]protoimpl.MessageInfo, 2)
var file_protos_edge_pinhole_v1_pinhole_proto_goTypes = []interface{}{
	(*SignRequest)(nil),           // 0: protos.edge.pinhole.v1.SignRequest
	(*SignResponse)(nil),          // 1: protos.edge.pinhole.v1.SignResponse
	(*timestamppb.Timestamp)(nil), // 2: google.protobuf.Timestamp
}
var file_protos_edge_pinhole_v1_pinhole_proto_depIdxs = []int32{
	2, // 0: protos.edge.pinhole.v1.SignResponse.expiration:type_name -> google.protobuf.Timestamp
	0, // 1: protos.edge.pinhole.v1.PinholeService.Sign:input_type -> protos.edge.pinhole.v1.SignRequest
	1, // 2: protos.edge.pinhole.v1.PinholeService.Sign:output_type -> protos.edge.pinhole.v1.SignResponse
	2, // [2:3] is the sub-list for method output_type
	1, // [1:2] is the sub-list for method input_type
	1, // [1:1] is the sub-list for extension type_name
	1, // [1:1] is the sub-list for extension extendee
	0, // [0:1] is the sub-list for field type_name
}

func init() { file_protos_edge_pinhole_v1_pinhole_proto_init() }
func file_protos_edge_pinhole_v1_pinhole_proto_init() {
	if File_protos_edge_pinhole_v1_pinhole_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_protos_edge_pinhole_v1_pinhole_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SignRequest); i {
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
		file_protos_edge_pinhole_v1_pinhole_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SignResponse); i {
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
			RawDescriptor: file_protos_edge_pinhole_v1_pinhole_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   2,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_protos_edge_pinhole_v1_pinhole_proto_goTypes,
		DependencyIndexes: file_protos_edge_pinhole_v1_pinhole_proto_depIdxs,
		MessageInfos:      file_protos_edge_pinhole_v1_pinhole_proto_msgTypes,
	}.Build()
	File_protos_edge_pinhole_v1_pinhole_proto = out.File
	file_protos_edge_pinhole_v1_pinhole_proto_rawDesc = nil
	file_protos_edge_pinhole_v1_pinhole_proto_goTypes = nil
	file_protos_edge_pinhole_v1_pinhole_proto_depIdxs = nil
}
