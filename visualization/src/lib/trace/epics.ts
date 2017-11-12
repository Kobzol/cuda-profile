import {Action} from 'redux';
import {combineEpics, ActionsObservable} from 'redux-observable';
import 'rxjs/add/operator/catch';
import 'rxjs/add/operator/map';
import '../util/redux-observable';
import {buildProfile} from '../profile/actions';
import {parseProfile} from '../profile/parse';


const loadTraceFileEpic = (action$: ActionsObservable<Action>) =>
    action$
        .ofAction(buildProfile.started)
        .map(action => {
            try
            {
                return buildProfile.done({
                    result: parseProfile(action.payload),
                    params: action.payload
                });
            }
            catch (error)
            {
                return buildProfile.failed({
                    error,
                    params: action.payload
                });
            }
        });

export const traceEpics = combineEpics(loadTraceFileEpic);
