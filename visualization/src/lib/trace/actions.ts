import {actionCreatorFactory} from 'typescript-fsa';

const actionCreator = actionCreatorFactory('parse');

export const loadTraceFile = actionCreator.async<File, {}, {}>('load-file');
