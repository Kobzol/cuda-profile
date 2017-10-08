import {Run} from './run';
import {Kernel} from './kernel';

export interface Profile
{
    run: Run;
    kernels: Kernel[];
}
