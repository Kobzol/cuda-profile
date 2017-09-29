import {Action} from 'redux';
import {combineEpics, ActionsObservable} from 'redux-observable';
import 'rxjs/add/observable/of';
import 'rxjs/add/operator/catch';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/mergeMap';
import 'typescript-fsa-redux-observable';
import {buildProfile as buildProfileAction} from './actions';
import {buildProfile} from './kernel';


const loadTraceFileEpic = (action$: ActionsObservable<Action>) =>
    action$
        .ofAction(buildProfileAction.started)
        .map(action => buildProfileAction.done({
            params: action.payload,
            result: buildProfile(action.payload)
        }));

export const traceEpics = combineEpics(loadTraceFileEpic);
