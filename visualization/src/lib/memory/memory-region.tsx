import * as React from 'react';
import {PureComponent} from "react";
import {MemoryAllocation} from "../data/memory-allocation";

interface Props
{
    allocation: MemoryAllocation;
}

export class MemoryRegion extends PureComponent<Props>
{
    render()
    {
        return (
            <svg width={100} height={20} viewBox="0,0,100,20">
                <rect width={100} height={20} fill="#0000FF" />
            </svg>
        );
    }
}
