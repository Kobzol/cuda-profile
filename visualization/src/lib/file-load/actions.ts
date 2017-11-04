import {actionCreatorFactory} from 'typescript-fsa';
import {FileLoadData} from './file';

const actionCreator = actionCreatorFactory('file-load');

export const loadFile = actionCreator.async<File, FileLoadData>('load');
export const resetFiles = actionCreator('reset');
