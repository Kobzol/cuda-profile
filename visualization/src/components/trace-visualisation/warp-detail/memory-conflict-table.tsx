import React, {PureComponent} from 'react';
import {Table} from 'reactstrap';
import {coalesceConflicts, getConflicts, Warp, WarpConflict} from '../../../lib/profile/warp';
import {formatAccessType, formatDim3} from '../../../lib/util/format';
import {Trace} from '../../../lib/profile/trace';
import {AddressRange} from '../../../lib/trace/selection';
import {addressAddStr, getAddressRangeSize} from '../../../lib/profile/address';

interface Props
{
    trace: Trace;
    warps: Warp[];
    onMemorySelect: (selection: AddressRange[]) => void;
}

export class MemoryConflictTable extends PureComponent<Props>
{
    render()
    {
        const conflicts = this.calculateConflicts(this.props.warps);
        if (conflicts.length === 0)
        {
            return <div>No memory conflicts detected in selected accesses.</div>;
        }

        return (
            <Table striped bordered hover>
                <thead>
                    <tr>
                        <th>Address</th>
                        <th>Threads</th>
                    </tr>
                </thead>
                {this.renderConflicts(conflicts)}
            </Table>
        );
    }
    renderConflicts = (conflicts: WarpConflict[]): JSX.Element =>
    {
        return (
            <tbody>
                {conflicts.map(conflict => this.renderConflict(conflict))}
            </tbody>
        );
    }
    renderConflict = (conflict: WarpConflict): JSX.Element =>
    {
        let threads = '';
        for (let i = 0; i < conflict.accesses.length; i++)
        {
            const access = conflict.accesses[i];
            const address = `${formatDim3(access.warp.blockIdx)}.${formatDim3(access.access.threadIdx)}`;
            threads += `${address} (${formatAccessType(access.warp.accessType)})`;

            if (i !== conflict.accesses.length - 1)
            {
                threads += ', ';
            }
        }

        const hashRange = (range: AddressRange) => `${range.from}-${range.to}`;
        const getRangeExtent = (range: AddressRange): string => {
            const next = addressAddStr(range.from, 1);
            if (range.to === next) return range.from;

            const byteCount = getAddressRangeSize(range);
            return `${range.from} - ${addressAddStr(range.to, -1)} (${byteCount} bytes)`;
        };

        return (
            <tr key={hashRange(conflict.address)}
                onMouseEnter={() => this.selectConflict(conflict)}
                onMouseLeave={() => this.selectConflict(null)}>
                <td>{getRangeExtent(conflict.address)}</td>
                <td>{threads}</td>
            </tr>
        );
    }

    calculateConflicts = (warps: Warp[]): WarpConflict[] =>
    {
        return coalesceConflicts(getConflicts(warps));
    }

    selectConflict = (conflict: WarpConflict | null) =>
    {
        this.props.onMemorySelect(conflict === null ? [] : [conflict.address]);
    }
}
