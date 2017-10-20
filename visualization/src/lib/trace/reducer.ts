import {reducerWithInitialState} from 'typescript-fsa-reducers';
import {buildProfile, selectAccessGroup, selectTrace} from './actions';
import {Profile} from '../profile/profile';
import {TraceSelection} from './trace-selection';
import {createSelector} from 'reselect';
import {AppState} from '../state/reducers';
import {Kernel} from '../profile/kernel';
import {Trace} from '../profile/trace';

export interface TraceState
{
    profile?: Profile;
    selectedTrace?: TraceSelection;
    selectedAccessGroup?: number;
}

const reducer = reducerWithInitialState<TraceState>({
    profile: {
        run: {
            start: 1,
            end: 200
        },
        kernels: [{
            metadata: {
                typeMap: ["", "", ""],
                locations: []
            },
            name: "",
            traces: [{
                gridDimension: {x: 1, y: 1, z: 1},
                blockDimension: {x: 1, y: 1, z: 1},
                start: 1,
                end: 200,
                allocations: [{
                    address: "0x1000",
                    size: 30 * 1024 * 1024,
                    elementSize: 1,
                    space: 1,
                    typeString: "",
                    typeIndex: 1,
                    active: true
                },{
                    address: "0x2000",
                    size: 100 * 1024 * 1024,
                    elementSize: 1,
                    space: 1,
                    typeString: "",
                    typeIndex: 1,
                    active: true
                }],
                accessGroups: [{
                    size: 0,
                    kind: 1,
                    space: 1,
                    typeIndex: 1,
                    timestamp: 10,
                    debugId: 1,
                    accesses: [{
                        id: 0,
                        address: "0x0111",
                        threadIdx: {
                            x: 1,
                            y: 1,
                            z: 1
                        },
                        blockIdx: {
                            x: 1,
                            y: 1,
                            z: 1
                        },
                        warpId: 1
                    }]
                }]
            }]
        }]
    },
    selectedTrace: null,
    selectedAccessGroup: null
})/*.case(buildProfile.done, (state, payload) => ({
    ...state,
    profile: payload.result,
    selectedTrace: null,
    selectedAccessGroup: null
}))*/.case(selectTrace, (state, payload) => ({
    ...state,
    selectedTrace: payload,
    selectedAccessGroup: null
})).case(selectAccessGroup, (state, payload) => ({
    ...state,
    selectedAccessGroup: payload
}));

export const selectedKernel = createSelector(
    (state: AppState) => state.trace,
    (state: TraceState) => state.selectedTrace !== null ? state.profile.kernels[state.selectedTrace.kernel] : null
);
export const selectedTrace = createSelector(
    (state: AppState) => state.trace,
    selectedKernel,
    (state: TraceState, kernel: Kernel) => kernel !== null ? kernel.traces[state.selectedTrace.trace] : null
);
export const selectedAccessGroup = createSelector(
    (state: AppState) => state.trace,
    selectedTrace,
    (state: TraceState, trace: Trace) => {
        if (state.selectedAccessGroup !== null &&
            state.selectedAccessGroup >= 0 &&
            state.selectedAccessGroup < trace.accessGroups.length)
        {
            return trace.accessGroups[state.selectedAccessGroup];
        }

        return null;
    }
);

export const traceReducer = reducer;
