import React, {PureComponent} from 'react';
import {MemoryAllocation} from '../../../lib/profile/memory-allocation';
import {MemoryBlock} from './memory-block/memory-block';
import {AddressRange, WarpAddressSelection} from '../../../lib/trace/selection';
import {MemoryMinimap} from './memory-minimap/memory-minimap';

import style from './memory-list.scss';

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
            <div>
                <h3>Memory allocations</h3>
                <div>
                    {this.props.allocations.map(alloc =>
                        <div
                            key={alloc.address}
                            className={style.blockRow}>
                            <MemoryBlock
                                allocation={alloc}
                                rangeSelections={this.props.rangeSelections}
                                onMemorySelect={this.props.onMemorySelect} />
                            <MemoryMinimap
                                width={200}
                                height={100}
                                rangeSelections={this.props.rangeSelections}
                                allocation={alloc} />
                        </div>
                    )}
                </div>
            </div>
        );
    }
}
