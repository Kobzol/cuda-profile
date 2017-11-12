import {Action} from 'redux';
import {Observable} from 'rxjs/Observable';
import {combineEpics, ActionsObservable} from 'redux-observable';
import 'rxjs/add/operator/map';
import 'rxjs/add/observable/concat';
import 'rxjs/add/observable/of';
import '../util/redux-observable';
import {buildProfile} from './actions';
import {push} from 'react-router-redux';
import {Routes} from '../nav/routes';
import {resetFiles} from '../file-load/actions';


const goToVisualisationAfterLoad = (action$: ActionsObservable<Action>) =>
    action$
        .ofAction(buildProfile.done)
        .flatMap(action => Observable.concat(
            Observable.of(push(Routes.TraceVisualisation)),
            Observable.of(resetFiles())
        ));

export const profileEpics = combineEpics(goToVisualisationAfterLoad);
