import {Action} from 'redux';
import {combineEpics, ActionsObservable} from 'redux-observable';
import 'rxjs/add/observable/of';
import 'rxjs/add/operator/catch';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/mergeMap';
import 'typescript-fsa-redux-observable';
import {buildProfile} from './actions';
import {parseProfile} from '../profile/parse';


const loadTraceFileEpic = (action$: ActionsObservable<Action>) =>
    action$
        .ofAction(buildProfile.started)
        .map(action => buildProfile.done({
            params: action.payload,
            result: parseProfile(action.payload)
        }));

export const traceEpics = combineEpics(loadTraceFileEpic);
