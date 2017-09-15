import {applyMiddleware, combineReducers, compose, createStore} from 'redux';
import createHistory from 'history/createBrowserHistory';
import {routerMiddleware, routerReducer} from 'react-router-redux';
import {createLogger} from 'redux-logger';
import {reducers} from './reducers';
import {rootEpic} from "./epics";
import {createEpicMiddleware} from "redux-observable";

export const history = createHistory();

const composeEnhancers = window['__REDUX_DEVTOOLS_EXTENSION_COMPOSE__'] || compose;
const router = routerMiddleware(history);
const epic = createEpicMiddleware(rootEpic);
const logger = createLogger();

export const store = createStore(
    combineReducers({
        ...reducers,
        router: routerReducer
    }),
    composeEnhancers(applyMiddleware(router, epic, logger))
);
