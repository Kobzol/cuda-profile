import React from 'react';
import {Kernel} from '../../lib/trace/kernel';

interface Props
{
    kernel: Kernel;
}

export function KernelComponent({kernel}: Props): JSX.Element
{
    return (
        <div>{kernel.metadata.kernel}</div>
    );
}
