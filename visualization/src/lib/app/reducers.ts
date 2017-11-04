import {REHYDRATE} from 'redux-persist/constants';
import {Action} from 'redux';

export interface AppState
{
    loaded: boolean;
}

const initialState: AppState = {
    loaded: false
};

function reducer(state: AppState = initialState, action: Action): AppState
{
    if (action.type === REHYDRATE)
    {
        return {
            ...state,
            loaded: true
        };
    }

    return state;
}

export const appReducer = reducer;
