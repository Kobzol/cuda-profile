import {reducerWithInitialState} from 'typescript-fsa-reducers';
import {buildKernels} from './actions';
import {Kernel} from './kernel';

export interface TraceState
{
    kernels: Kernel[];
    selectedKernel?: number;
    buildingKernels: boolean;
}

const reducer = reducerWithInitialState<TraceState>({
    kernels: [],
    selectedKernel: null,
    buildingKernels: false
}).case(buildKernels.started, state => ({
    ...state,
    buildingKernels: true
}))
    .case(buildKernels.done, (state, payload) => ({
    ...state,
    kernels: payload.result,
    buildingKernels: false
}));

export const traceReducer = reducer;
