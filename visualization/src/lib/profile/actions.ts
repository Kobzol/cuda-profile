import {actionCreatorFactory} from 'typescript-fsa';
import {TraceFile} from '../file-load/file';
import {Profile} from './profile';

const actionCreator = actionCreatorFactory('profile');

export const buildProfile = actionCreator.async<TraceFile[], Profile>('build');
