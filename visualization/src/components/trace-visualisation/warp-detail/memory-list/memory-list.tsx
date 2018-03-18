import React, {PureComponent} from 'react';
import {MemoryAllocation} from '../../../../lib/profile/memory-allocation';
import {AllocationView} from './allocation-view/allocation-view';
import {AddressRange, WarpAccess} from '../../../../lib/trace/selection';
import styled from 'styled-components';
import {Warp} from '../../../../lib/profile/warp';
import {any} from 'ramda';
import {getAccessesAddressRange, getAllocationAddressRange, intersects} from '../../../../lib/profile/address';

interface Props
{
    allocations: MemoryAllocation[];
    selectedAccesses: WarpAccess[];
    selectedWarps: Warp[];
    onMemorySelect(memorySelection: AddressRange[]): void;
}

const Row = styled.div`
  display: flex;
`;

export class MemoryList extends PureComponent<Props>
{
    render(): JSX.Element
    {
        /*<MemoryMinimap
                            width={200}
                            height={100}
                            rangeSelections={this.props.rangeSelections}
                            allocation={alloc} />
        */
        const allocations = this.getActiveAllocations(this.props.selectedWarps, this.props.allocations);
        return (
            <div>
                {allocations.length === 0 && <div>No allocations detected for selected accesses.</div>}
                {allocations.map(alloc =>
                    <Row key={alloc.address}>
                        <AllocationView
                            allocation={alloc}
                            selectedAccesses={this.props.selectedAccesses}
                            selectedWarps={this.props.selectedWarps}
                            onMemorySelect={this.props.onMemorySelect} />
                    </Row>
                )}
            </div>
        );
    }

    getActiveAllocations = (warps: Warp[], allocations: MemoryAllocation[]): MemoryAllocation[] =>
    {
        return allocations.filter(alloc =>
            any(w => {
                const warpRange = getAccessesAddressRange(w.accesses, w.size);
                const allocRange = getAllocationAddressRange(alloc);
                return intersects(allocRange, warpRange);
            }, warps)
        );
    }
}
