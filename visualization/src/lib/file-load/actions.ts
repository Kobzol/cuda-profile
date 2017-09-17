import {actionCreatorFactory} from 'typescript-fsa';
import {FileLoadData} from "./trace-file";

const actionCreator = actionCreatorFactory('file-load');

export const loadFile = actionCreator.async<File, FileLoadData>('load');
