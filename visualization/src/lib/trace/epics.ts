import {Action} from 'redux';
import {combineEpics, ActionsObservable} from 'redux-observable';
import 'rxjs/add/observable/of';
import 'rxjs/add/operator/catch';
import 'rxjs/add/operator/map';
import 'rxjs/add/operator/mergeMap';
import 'typescript-fsa-redux-observable';
import {buildKernels as buildKernelsAction} from './actions';
import {buildKernels} from './kernel';


const loadTraceFileEpic = (action$: ActionsObservable<Action>) =>
    action$
        .ofAction(buildKernelsAction.started)
        .map(action => buildKernelsAction.done({
            params: action.payload,
            result: buildKernels(action.payload)
        }));

export const traceEpics = combineEpics(loadTraceFileEpic);
