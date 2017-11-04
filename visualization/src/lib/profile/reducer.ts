import {reducerWithInitialState} from 'typescript-fsa-reducers';
import {buildProfile} from './actions';
import {Profile} from './profile';

export interface ProfileState
{
    profile?: Profile;
    buildError: string;
}

const reducer = reducerWithInitialState<ProfileState>({
    profile: null,
    buildError: ''
}).case(buildProfile.failed, (state, payload) => ({
    ...state,
    buildError: payload.error.message
})).case(buildProfile.done, (state, payload) => ({
    ...state,
    profile: payload.result,
    buildError: ''
}));

export const profileReducer = reducer;
