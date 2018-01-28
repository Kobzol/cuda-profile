import React, {PureComponent} from 'react';
import {AddressSpace, Warp} from '../../../../lib/profile/warp';
import {Trace} from '../../../../lib/profile/trace';
import {AddressRange} from '../../../../lib/trace/selection';
import * as _ from 'lodash';
import {MemoryAccess} from '../../../../lib/profile/memory-access';
import {addressToNum, getAccessAddressRange} from '../../../../lib/profile/address';
import {formatDim3} from '../../../../lib/util/format';
import {Table} from 'react-bootstrap';
import {createSelector, Selector} from 'reselect';
import classNames from 'classnames';

import style from './bank-conflict-table.scss';

interface State
{
    getAccessMap: Selector<Warp[], AccessMap>;
}
interface Props
{
    trace: Trace;
    warps: Warp[];
    onMemorySelect: (selection: AddressRange[]) => void;
}

interface AccessMap
{
    [bank: number]: {access: MemoryAccess, warp: Warp}[];
}

export class BankConflictTable extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            getAccessMap: createSelector(warps => warps, warps => this.createAccessMap(this.getSharedWarps(warps)))
        };
    }

    render()
    {
        if (this.getSharedWarps(this.props.warps).length === 0)
        {
            return <div>No accesses to shared memory in selected warps.</div>;
        }

        const accessMap = this.state.getAccessMap(this.props.warps);
        return (
            <Table striped bordered condensed hover>
                <thead>
                    <tr>
                        {Object.keys(accessMap).map(bank => {
                            const conflict = accessMap[bank].length > 1;
                            const title = `Bank #${bank} ${conflict ? '(conflict)' : ''}`;

                            return (
                                <th onMouseEnter={() => this.selectBank(parseInt(bank, 10))}
                                    onMouseLeave={() => this.selectBank(null)}
                                    key={bank}
                                    title={title}
                                    className={classNames(style.bankHeader, {
                                        [style.conflicted] : conflict
                                    })}>
                                    {bank}
                                </th>
                            );
                        })}
                    </tr>
                </thead>
                {this.renderRows(accessMap)}
            </Table>
        );
    }
    renderRows = (accessMap: AccessMap): JSX.Element =>
    {
        const rows: JSX.Element[] = [];
        const validColumns = Object.keys(accessMap);

        let index = 0;
        while (true)
        {
            const columns = validColumns.map(col => index < accessMap[col].length ? accessMap[col][index] : null);
            if (_.every(columns, col => col === null)) break;

            rows.push(
                <tr key={index}>
                    {columns.map((col, i) =>
                        <td onMouseEnter={() => this.selectBank(parseInt(validColumns[i], 10))}
                            onMouseLeave={() => this.selectBank(null)}
                            key={i}>
                            {col !== null && this.renderAccess(col.warp, col.access)}
                        </td>
                    )}
                </tr>
            );

            index++;
        }

        return <tbody>{rows}</tbody>;
    }

    renderBank = (accessMap: AccessMap, bank: number): JSX.Element =>
    {
        const accesses = accessMap[bank];

        return (
            <div key={bank}>
                <div>Bank #{bank}</div>
                <div>
                    {accesses !== undefined && accesses.map(pair => this.renderAccess(pair.warp, pair.access))}
                </div>
            </div>
        );
    }
    renderAccess = (warp: Warp, access: MemoryAccess): JSX.Element =>
    {
        return (
            <div key={access.id}
                 title={`${formatDim3(warp.blockIdx)}.${formatDim3(access.threadIdx)} at ${access.address}`}
                 className={classNames(style.access, style.conflicted)} />
        );
    }

    selectBank = (bank: number | null) =>
    {
        if (bank === null)
        {
            this.props.onMemorySelect([]);
        }
        else
        {
            const accesses = this.state.getAccessMap(this.props.warps)[bank] || [];
            const ranges = accesses.map(pair => getAccessAddressRange(pair.access, pair.warp.size));
            this.props.onMemorySelect(ranges);
        }
    }

    getSharedWarps = (warps: Warp[]): Warp[] =>
    {
        return warps.filter(warp => warp.space === AddressSpace.Shared);
    }
    createAccessMap = (warps: Warp[]): AccessMap =>
    {
        const accessesWithWarp = _.flatMap(warps, warp => warp.accesses.map(access => ({
            warp,
            access
        })));

        return _.groupBy(accessesWithWarp, (pair: {warp: Warp, access: MemoryAccess}) =>
            addressToNum(pair.access.address).divide(4).mod(32).toJSNumber()
        );
    }
}
