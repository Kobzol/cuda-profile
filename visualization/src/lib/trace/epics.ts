import {Action} from 'redux';
import {combineEpics, ActionsObservable} from 'redux-observable';
import 'rxjs/add/operator/catch';
import 'rxjs/add/operator/map';
import 'rxjs/add/observable/of';
import '../util/redux-observable';
import {buildProfile} from '../profile/actions';
import {parseProfileAsync} from '../profile/parse';
import {Observable} from 'rxjs/Observable';


const loadTraceFileEpic = (action$: ActionsObservable<Action>) =>
    action$
        .ofAction(buildProfile.started)
        .flatMap(action =>
            parseProfileAsync(action.payload)
                .map(result => buildProfile.done({
                    result,
                    params: action.payload
                }))
                .catch(error => Observable.of(buildProfile.failed({
                    error,
                    params: action.payload
                })))
        );

export const traceEpics = combineEpics(loadTraceFileEpic);
