// Code generated by counterfeiter. DO NOT EDIT.
package fake

import (
	"context"
	"sync"

	"github.com/aws/aws-sdk-go-v2/service/s3"
)

type S3Client struct {
	GetObjectStub        func(context.Context, *s3.GetObjectInput, ...func(*s3.Options)) (*s3.GetObjectOutput, error)
	getObjectMutex       sync.RWMutex
	getObjectArgsForCall []struct {
		arg1 context.Context
		arg2 *s3.GetObjectInput
		arg3 []func(*s3.Options)
	}
	getObjectReturns struct {
		result1 *s3.GetObjectOutput
		result2 error
	}
	getObjectReturnsOnCall map[int]struct {
		result1 *s3.GetObjectOutput
		result2 error
	}
	HeadObjectStub        func(context.Context, *s3.HeadObjectInput, ...func(*s3.Options)) (*s3.HeadObjectOutput, error)
	headObjectMutex       sync.RWMutex
	headObjectArgsForCall []struct {
		arg1 context.Context
		arg2 *s3.HeadObjectInput
		arg3 []func(*s3.Options)
	}
	headObjectReturns struct {
		result1 *s3.HeadObjectOutput
		result2 error
	}
	headObjectReturnsOnCall map[int]struct {
		result1 *s3.HeadObjectOutput
		result2 error
	}
	PutObjectStub        func(context.Context, *s3.PutObjectInput, ...func(*s3.Options)) (*s3.PutObjectOutput, error)
	putObjectMutex       sync.RWMutex
	putObjectArgsForCall []struct {
		arg1 context.Context
		arg2 *s3.PutObjectInput
		arg3 []func(*s3.Options)
	}
	putObjectReturns struct {
		result1 *s3.PutObjectOutput
		result2 error
	}
	putObjectReturnsOnCall map[int]struct {
		result1 *s3.PutObjectOutput
		result2 error
	}
	invocations      map[string][][]interface{}
	invocationsMutex sync.RWMutex
}

func (fake *S3Client) GetObject(arg1 context.Context, arg2 *s3.GetObjectInput, arg3 ...func(*s3.Options)) (*s3.GetObjectOutput, error) {
	fake.getObjectMutex.Lock()
	ret, specificReturn := fake.getObjectReturnsOnCall[len(fake.getObjectArgsForCall)]
	fake.getObjectArgsForCall = append(fake.getObjectArgsForCall, struct {
		arg1 context.Context
		arg2 *s3.GetObjectInput
		arg3 []func(*s3.Options)
	}{arg1, arg2, arg3})
	stub := fake.GetObjectStub
	fakeReturns := fake.getObjectReturns
	fake.recordInvocation("GetObject", []interface{}{arg1, arg2, arg3})
	fake.getObjectMutex.Unlock()
	if stub != nil {
		return stub(arg1, arg2, arg3...)
	}
	if specificReturn {
		return ret.result1, ret.result2
	}
	return fakeReturns.result1, fakeReturns.result2
}

func (fake *S3Client) GetObjectCallCount() int {
	fake.getObjectMutex.RLock()
	defer fake.getObjectMutex.RUnlock()
	return len(fake.getObjectArgsForCall)
}

func (fake *S3Client) GetObjectCalls(stub func(context.Context, *s3.GetObjectInput, ...func(*s3.Options)) (*s3.GetObjectOutput, error)) {
	fake.getObjectMutex.Lock()
	defer fake.getObjectMutex.Unlock()
	fake.GetObjectStub = stub
}

func (fake *S3Client) GetObjectArgsForCall(i int) (context.Context, *s3.GetObjectInput, []func(*s3.Options)) {
	fake.getObjectMutex.RLock()
	defer fake.getObjectMutex.RUnlock()
	argsForCall := fake.getObjectArgsForCall[i]
	return argsForCall.arg1, argsForCall.arg2, argsForCall.arg3
}

func (fake *S3Client) GetObjectReturns(result1 *s3.GetObjectOutput, result2 error) {
	fake.getObjectMutex.Lock()
	defer fake.getObjectMutex.Unlock()
	fake.GetObjectStub = nil
	fake.getObjectReturns = struct {
		result1 *s3.GetObjectOutput
		result2 error
	}{result1, result2}
}

func (fake *S3Client) GetObjectReturnsOnCall(i int, result1 *s3.GetObjectOutput, result2 error) {
	fake.getObjectMutex.Lock()
	defer fake.getObjectMutex.Unlock()
	fake.GetObjectStub = nil
	if fake.getObjectReturnsOnCall == nil {
		fake.getObjectReturnsOnCall = make(map[int]struct {
			result1 *s3.GetObjectOutput
			result2 error
		})
	}
	fake.getObjectReturnsOnCall[i] = struct {
		result1 *s3.GetObjectOutput
		result2 error
	}{result1, result2}
}

func (fake *S3Client) HeadObject(arg1 context.Context, arg2 *s3.HeadObjectInput, arg3 ...func(*s3.Options)) (*s3.HeadObjectOutput, error) {
	fake.headObjectMutex.Lock()
	ret, specificReturn := fake.headObjectReturnsOnCall[len(fake.headObjectArgsForCall)]
	fake.headObjectArgsForCall = append(fake.headObjectArgsForCall, struct {
		arg1 context.Context
		arg2 *s3.HeadObjectInput
		arg3 []func(*s3.Options)
	}{arg1, arg2, arg3})
	stub := fake.HeadObjectStub
	fakeReturns := fake.headObjectReturns
	fake.recordInvocation("HeadObject", []interface{}{arg1, arg2, arg3})
	fake.headObjectMutex.Unlock()
	if stub != nil {
		return stub(arg1, arg2, arg3...)
	}
	if specificReturn {
		return ret.result1, ret.result2
	}
	return fakeReturns.result1, fakeReturns.result2
}

func (fake *S3Client) HeadObjectCallCount() int {
	fake.headObjectMutex.RLock()
	defer fake.headObjectMutex.RUnlock()
	return len(fake.headObjectArgsForCall)
}

func (fake *S3Client) HeadObjectCalls(stub func(context.Context, *s3.HeadObjectInput, ...func(*s3.Options)) (*s3.HeadObjectOutput, error)) {
	fake.headObjectMutex.Lock()
	defer fake.headObjectMutex.Unlock()
	fake.HeadObjectStub = stub
}

func (fake *S3Client) HeadObjectArgsForCall(i int) (context.Context, *s3.HeadObjectInput, []func(*s3.Options)) {
	fake.headObjectMutex.RLock()
	defer fake.headObjectMutex.RUnlock()
	argsForCall := fake.headObjectArgsForCall[i]
	return argsForCall.arg1, argsForCall.arg2, argsForCall.arg3
}

func (fake *S3Client) HeadObjectReturns(result1 *s3.HeadObjectOutput, result2 error) {
	fake.headObjectMutex.Lock()
	defer fake.headObjectMutex.Unlock()
	fake.HeadObjectStub = nil
	fake.headObjectReturns = struct {
		result1 *s3.HeadObjectOutput
		result2 error
	}{result1, result2}
}

func (fake *S3Client) HeadObjectReturnsOnCall(i int, result1 *s3.HeadObjectOutput, result2 error) {
	fake.headObjectMutex.Lock()
	defer fake.headObjectMutex.Unlock()
	fake.HeadObjectStub = nil
	if fake.headObjectReturnsOnCall == nil {
		fake.headObjectReturnsOnCall = make(map[int]struct {
			result1 *s3.HeadObjectOutput
			result2 error
		})
	}
	fake.headObjectReturnsOnCall[i] = struct {
		result1 *s3.HeadObjectOutput
		result2 error
	}{result1, result2}
}

func (fake *S3Client) PutObject(arg1 context.Context, arg2 *s3.PutObjectInput, arg3 ...func(*s3.Options)) (*s3.PutObjectOutput, error) {
	fake.putObjectMutex.Lock()
	ret, specificReturn := fake.putObjectReturnsOnCall[len(fake.putObjectArgsForCall)]
	fake.putObjectArgsForCall = append(fake.putObjectArgsForCall, struct {
		arg1 context.Context
		arg2 *s3.PutObjectInput
		arg3 []func(*s3.Options)
	}{arg1, arg2, arg3})
	stub := fake.PutObjectStub
	fakeReturns := fake.putObjectReturns
	fake.recordInvocation("PutObject", []interface{}{arg1, arg2, arg3})
	fake.putObjectMutex.Unlock()
	if stub != nil {
		return stub(arg1, arg2, arg3...)
	}
	if specificReturn {
		return ret.result1, ret.result2
	}
	return fakeReturns.result1, fakeReturns.result2
}

func (fake *S3Client) PutObjectCallCount() int {
	fake.putObjectMutex.RLock()
	defer fake.putObjectMutex.RUnlock()
	return len(fake.putObjectArgsForCall)
}

func (fake *S3Client) PutObjectCalls(stub func(context.Context, *s3.PutObjectInput, ...func(*s3.Options)) (*s3.PutObjectOutput, error)) {
	fake.putObjectMutex.Lock()
	defer fake.putObjectMutex.Unlock()
	fake.PutObjectStub = stub
}

func (fake *S3Client) PutObjectArgsForCall(i int) (context.Context, *s3.PutObjectInput, []func(*s3.Options)) {
	fake.putObjectMutex.RLock()
	defer fake.putObjectMutex.RUnlock()
	argsForCall := fake.putObjectArgsForCall[i]
	return argsForCall.arg1, argsForCall.arg2, argsForCall.arg3
}

func (fake *S3Client) PutObjectReturns(result1 *s3.PutObjectOutput, result2 error) {
	fake.putObjectMutex.Lock()
	defer fake.putObjectMutex.Unlock()
	fake.PutObjectStub = nil
	fake.putObjectReturns = struct {
		result1 *s3.PutObjectOutput
		result2 error
	}{result1, result2}
}

func (fake *S3Client) PutObjectReturnsOnCall(i int, result1 *s3.PutObjectOutput, result2 error) {
	fake.putObjectMutex.Lock()
	defer fake.putObjectMutex.Unlock()
	fake.PutObjectStub = nil
	if fake.putObjectReturnsOnCall == nil {
		fake.putObjectReturnsOnCall = make(map[int]struct {
			result1 *s3.PutObjectOutput
			result2 error
		})
	}
	fake.putObjectReturnsOnCall[i] = struct {
		result1 *s3.PutObjectOutput
		result2 error
	}{result1, result2}
}

func (fake *S3Client) Invocations() map[string][][]interface{} {
	fake.invocationsMutex.RLock()
	defer fake.invocationsMutex.RUnlock()
	fake.getObjectMutex.RLock()
	defer fake.getObjectMutex.RUnlock()
	fake.headObjectMutex.RLock()
	defer fake.headObjectMutex.RUnlock()
	fake.putObjectMutex.RLock()
	defer fake.putObjectMutex.RUnlock()
	copiedInvocations := map[string][][]interface{}{}
	for key, value := range fake.invocations {
		copiedInvocations[key] = value
	}
	return copiedInvocations
}

func (fake *S3Client) recordInvocation(key string, args []interface{}) {
	fake.invocationsMutex.Lock()
	defer fake.invocationsMutex.Unlock()
	if fake.invocations == nil {
		fake.invocations = map[string][][]interface{}{}
	}
	if fake.invocations[key] == nil {
		fake.invocations[key] = [][]interface{}{}
	}
	fake.invocations[key] = append(fake.invocations[key], args)
}
