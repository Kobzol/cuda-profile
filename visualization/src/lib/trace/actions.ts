import {actionCreatorFactory} from 'typescript-fsa';

const actionCreator = actionCreatorFactory('parse');

export const loadFile = actionCreator.async<File, {}, {}>('load-file');
