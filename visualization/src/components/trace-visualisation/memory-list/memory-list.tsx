import React, {PureComponent} from 'react';
import {MemoryAllocation} from '../../../lib/profile/memory-allocation';
import {MemoryBlock} from '../memory-block/memory-block';
import {AddressRange, WarpAddressSelection} from '../../../lib/trace/selection';

interface Props
{
    allocations: MemoryAllocation[];
    rangeSelections: WarpAddressSelection[];
    onMemorySelect: (memorySelection: AddressRange) => void;
}

export class MemoryList extends PureComponent<Props>
{
    render()
    {
        return (
            <div className='memory-block-wrapper'>
                <h3>Memory allocations</h3>
                <div>
                    {this.props.allocations.map(alloc =>
                        <MemoryBlock
                            key={alloc.address}
                            allocation={alloc}
                            rangeSelections={this.props.rangeSelections}
                            onMemorySelect={this.props.onMemorySelect} />
                    )}
                </div>
            </div>
        );
    }
}
