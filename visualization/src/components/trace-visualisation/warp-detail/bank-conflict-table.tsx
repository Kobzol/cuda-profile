import React, {PureComponent} from 'react';
import {AddressSpace, Warp} from '../../../lib/profile/warp';
import {Trace} from '../../../lib/profile/trace';
import {AddressRange} from '../../../lib/trace/selection';
import {MemoryAccess} from '../../../lib/profile/memory-access';
import {addressToNum, getAccessAddressRange} from '../../../lib/profile/address';
import {formatDim3} from '../../../lib/util/format';
import {Alert, Table} from 'reactstrap';
import {createSelector, Selector} from 'reselect';
import MdWarning from 'react-icons/lib/md/warning';
import styled from 'styled-components';
import {all, chain, groupBy} from 'ramda';

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

const BankTable = styled(Table)`
  width: auto;
`;
interface BankHeaderProps {
    conflicted: boolean;
}
const BankHeader = styled.th`
  text-align: center;
  padding: 0 !important;
  ${(props: BankHeaderProps) => props.conflicted ? `
    background-color: #8F3938;
    color: #FFFFFF;
  ` : ''}
`;
const Access = styled.td`
  padding: 0.25rem !important;
  font-size: 11px;
  text-align: center;
`;

export class BankConflictTable extends PureComponent<Props, State>
{
    state: State = {
        getAccessMap: createSelector(warps => warps, warps => this.createAccessMap(this.getSharedWarps(warps)))
    };

    render()
    {
        const sharedWarps = this.getSharedWarps(this.props.warps);
        if (sharedWarps.length === 0)
        {
            return <div>No accesses to shared memory in selected accesses.</div>;
        }

        const accessMap = this.state.getAccessMap(this.props.warps);
        return (
            <div>
                <BankTable striped bordered hover>
                    <thead>
                        <tr>
                            {Object.keys(accessMap).map(bank => {
                                const conflict = (accessMap[bank] || []).length > 1;
                                const title = `Bank #${bank} ${conflict ? '(conflict)' : ''}`;

                                return (
                                    <BankHeader onMouseEnter={() => this.selectBank(parseInt(bank, 10))}
                                        onMouseLeave={() => this.selectBank(null)}
                                        key={bank}
                                        title={title}
                                        conflicted={conflict}>
                                        {bank}
                                    </BankHeader>
                                );
                            })}
                        </tr>
                    </thead>
                    {this.renderRows(accessMap)}
                </BankTable>
                {sharedWarps.length > 1 &&
                <Alert color='warning'>
                    <MdWarning /> More than one warp with shared memory access selected,
                    bank conflicts happen only within the context of a single warp.
                </Alert>}
            </div>
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
            if (all(col => col === null, columns)) break;

            rows.push(
                <tr key={index}>
                    {columns.map((col, i) =>
                        <Access onMouseEnter={() => this.selectBank(parseInt(validColumns[i], 10))}
                            onMouseLeave={() => this.selectBank(null)}
                            key={i}
                            title={col !== null ?
                            `${formatDim3(col.warp.blockIdx)}.${formatDim3(col.access.threadIdx)} ` +
                            `at ${col.access.address}` : ''}>
                            {col !== null && this.renderAccess(col.warp, col.access)}
                        </Access>
                    )}
                </tr>
            );

            index++;
        }

        return <tbody>{rows}</tbody>;
    }

    renderAccess = (warp: Warp, access: MemoryAccess): JSX.Element =>
    {
        return (
            <span>{formatDim3(access.threadIdx)}</span>
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
        const accessesWithWarp = chain(warp => warp.accesses.map(access => ({
            warp,
            access
        })), warps);

        return groupBy((pair: {warp: Warp, access: MemoryAccess}) =>
                addressToNum(pair.access.address)
                    .divide(this.props.trace.bankSize)
                    .mod(this.props.trace.warpSize)
                    .toJSNumber()
                    .toString(),
            accessesWithWarp);
    }
}
