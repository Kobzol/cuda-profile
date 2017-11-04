import {Action} from 'redux';
import {combineEpics, ActionsObservable} from 'redux-observable';
import 'rxjs/add/operator/map';
import 'typescript-fsa-redux-observable';
import {buildProfile} from './actions';
import {push} from 'react-router-redux';
import {Routes} from '../nav/routes';


const goToVisualisationAfterLoad = (action$: ActionsObservable<Action>) =>
    action$
        .ofAction(buildProfile.done)
        .map(action => push(Routes.TraceVisualisation));

export const profileEpics = combineEpics(goToVisualisationAfterLoad);
