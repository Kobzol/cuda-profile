import React, {PureComponent} from 'react';
import {MemoryAllocation} from '../../../../lib/profile/memory-allocation';
import {MemoryBlock} from './memory-block/memory-block';
import {AddressRange, WarpAddressSelection} from '../../../../lib/trace/selection';
import {MemoryMinimap} from './memory-minimap/memory-minimap';
import styled from 'styled-components';
import {Warp} from '../../../../lib/profile/warp';
import {any} from 'ramda';
import {getAccessesAddressRange, getAllocationAddressRange, intersects} from '../../../../lib/profile/address';

interface Props
{
    allocations: MemoryAllocation[];
    rangeSelections: WarpAddressSelection[];
    selectedWarps: Warp[];
    onMemorySelect(memorySelection: AddressRange[]): void;
}

const Row = styled.div`
  display: flex;
`;

export class MemoryList extends PureComponent<Props>
{
    render()
    {
        const allocations = this.getActiveAllocations(this.props.selectedWarps, this.props.allocations);
        return (
            <div>
                {allocations.length === 0 && <div>No allocations detected for selected accesses.</div>}
                {allocations.map(alloc =>
                    <Row key={alloc.address}>
                        <MemoryBlock
                            allocation={alloc}
                            rangeSelections={this.props.rangeSelections}
                            onMemorySelect={this.props.onMemorySelect} />
                        <MemoryMinimap
                            width={200}
                            height={100}
                            rangeSelections={this.props.rangeSelections}
                            allocation={alloc} />
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
