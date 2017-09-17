import * as React from 'react';
import {PureComponent} from "react";
import {MemoryAllocation} from "./memory-allocation";
import {MemoryRegion} from "./memory-region";

interface Props
{
    allocations: MemoryAllocation[];
}

export class MemoryMap extends PureComponent<Props>
{
    render()
    {
        const allocations: JSX.Element[] = [];
        for (const alloc of this.props.allocations)
        {
            allocations.push(
                <MemoryRegion key={alloc.address} allocation={alloc} />
            );
        }

        return (
            <div>
                {allocations}
            </div>
        );
    }
}
