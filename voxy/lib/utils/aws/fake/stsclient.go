// Code generated by counterfeiter. DO NOT EDIT.
package fake

import (
	"context"
	"sync"

	"github.com/aws/aws-sdk-go-v2/service/sts"
)

type STSClient struct {
	GetCallerIdentityStub        func(context.Context, *sts.GetCallerIdentityInput, ...func(*sts.Options)) (*sts.GetCallerIdentityOutput, error)
	getCallerIdentityMutex       sync.RWMutex
	getCallerIdentityArgsForCall []struct {
		arg1 context.Context
		arg2 *sts.GetCallerIdentityInput
		arg3 []func(*sts.Options)
	}
	getCallerIdentityReturns struct {
		result1 *sts.GetCallerIdentityOutput
		result2 error
	}
	getCallerIdentityReturnsOnCall map[int]struct {
		result1 *sts.GetCallerIdentityOutput
		result2 error
	}
	invocations      map[string][][]interface{}
	invocationsMutex sync.RWMutex
}

func (fake *STSClient) GetCallerIdentity(arg1 context.Context, arg2 *sts.GetCallerIdentityInput, arg3 ...func(*sts.Options)) (*sts.GetCallerIdentityOutput, error) {
	fake.getCallerIdentityMutex.Lock()
	ret, specificReturn := fake.getCallerIdentityReturnsOnCall[len(fake.getCallerIdentityArgsForCall)]
	fake.getCallerIdentityArgsForCall = append(fake.getCallerIdentityArgsForCall, struct {
		arg1 context.Context
		arg2 *sts.GetCallerIdentityInput
		arg3 []func(*sts.Options)
	}{arg1, arg2, arg3})
	stub := fake.GetCallerIdentityStub
	fakeReturns := fake.getCallerIdentityReturns
	fake.recordInvocation("GetCallerIdentity", []interface{}{arg1, arg2, arg3})
	fake.getCallerIdentityMutex.Unlock()
	if stub != nil {
		return stub(arg1, arg2, arg3...)
	}
	if specificReturn {
		return ret.result1, ret.result2
	}
	return fakeReturns.result1, fakeReturns.result2
}

func (fake *STSClient) GetCallerIdentityCallCount() int {
	fake.getCallerIdentityMutex.RLock()
	defer fake.getCallerIdentityMutex.RUnlock()
	return len(fake.getCallerIdentityArgsForCall)
}

func (fake *STSClient) GetCallerIdentityCalls(stub func(context.Context, *sts.GetCallerIdentityInput, ...func(*sts.Options)) (*sts.GetCallerIdentityOutput, error)) {
	fake.getCallerIdentityMutex.Lock()
	defer fake.getCallerIdentityMutex.Unlock()
	fake.GetCallerIdentityStub = stub
}

func (fake *STSClient) GetCallerIdentityArgsForCall(i int) (context.Context, *sts.GetCallerIdentityInput, []func(*sts.Options)) {
	fake.getCallerIdentityMutex.RLock()
	defer fake.getCallerIdentityMutex.RUnlock()
	argsForCall := fake.getCallerIdentityArgsForCall[i]
	return argsForCall.arg1, argsForCall.arg2, argsForCall.arg3
}

func (fake *STSClient) GetCallerIdentityReturns(result1 *sts.GetCallerIdentityOutput, result2 error) {
	fake.getCallerIdentityMutex.Lock()
	defer fake.getCallerIdentityMutex.Unlock()
	fake.GetCallerIdentityStub = nil
	fake.getCallerIdentityReturns = struct {
		result1 *sts.GetCallerIdentityOutput
		result2 error
	}{result1, result2}
}

func (fake *STSClient) GetCallerIdentityReturnsOnCall(i int, result1 *sts.GetCallerIdentityOutput, result2 error) {
	fake.getCallerIdentityMutex.Lock()
	defer fake.getCallerIdentityMutex.Unlock()
	fake.GetCallerIdentityStub = nil
	if fake.getCallerIdentityReturnsOnCall == nil {
		fake.getCallerIdentityReturnsOnCall = make(map[int]struct {
			result1 *sts.GetCallerIdentityOutput
			result2 error
		})
	}
	fake.getCallerIdentityReturnsOnCall[i] = struct {
		result1 *sts.GetCallerIdentityOutput
		result2 error
	}{result1, result2}
}

func (fake *STSClient) Invocations() map[string][][]interface{} {
	fake.invocationsMutex.RLock()
	defer fake.invocationsMutex.RUnlock()
	fake.getCallerIdentityMutex.RLock()
	defer fake.getCallerIdentityMutex.RUnlock()
	copiedInvocations := map[string][][]interface{}{}
	for key, value := range fake.invocations {
		copiedInvocations[key] = value
	}
	return copiedInvocations
}

func (fake *STSClient) recordInvocation(key string, args []interface{}) {
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