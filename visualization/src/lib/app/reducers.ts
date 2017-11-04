import {REHYDRATE} from 'redux-persist/lib/constants';
import {Action} from 'redux';

export interface AppState
{

}

const initialState: AppState = {

};

function reducer(state: AppState = initialState, action: Action): AppState
{
    return state;
}

export const appReducer = reducer;
