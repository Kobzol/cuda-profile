import React, {PureComponent} from 'react';
import {Kernel} from '../../../lib/profile/kernel';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {WarpFilter} from './warp-filter/warp-filter';
import {Dim3} from '../../../lib/profile/dim3';
import {WarpOverview} from './warp-overview/warp-overview';

import './warp-panel.scss';
import {Button} from 'react-bootstrap';

interface Props
{
    kernel: Kernel;
    trace: Trace;
    selectedWarps: Warp[];
    selectWarps: (warps: Warp[]) => void;
}

interface State
{
    blockFilter: Dim3;
}

export class WarpPanel extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            blockFilter: { x: null, y: null, z: null }
        };
    }

    render()
    {
        const warps = this.getFilteredWarps();

        return (
            <div className='warps-wrapper'>
                <h3>Warps</h3>
                <div>
                    {this.isFilterActive() ? `${warps.length} filtered (${this.props.trace.warps.length} total)` :
                        `${this.props.trace.warps.length} total`}
                </div>
                {this.isFilterActive() && <Button onClick={this.resetFilter}>Reset filter</Button>}
                <WarpFilter
                    label='Block'
                    filter={this.state.blockFilter}
                    onFilterChange={this.changeBlockFilter} />
                <WarpOverview
                    warps={warps}
                    selectedWarps={this.props.selectedWarps}
                    onWarpSelect={this.props.selectWarps} />
            </div>
        );
    }

    getFilteredWarps = (): Warp[] =>
    {
        const {x, y, z} = this.state.blockFilter;
        if (x === null && y === null && z === null) return this.props.trace.warps;

        return this.props.trace.warps.filter(warp => {
            if (x !== null && warp.blockIdx.x !== x) return false;
            if (y !== null && warp.blockIdx.y !== y) return false;
            return !(z !== null && warp.blockIdx.z !== z);
        });
    }

    isFilterActive = (): boolean =>
    {
        return (
            this.state.blockFilter.x !== null ||
            this.state.blockFilter.y !== null ||
            this.state.blockFilter.z !== null
        );
    }

    changeBlockFilter = (blockFilter: Dim3) =>
    {
        this.setState(() => ({
            blockFilter
        }));
    }
    resetFilter = () =>
    {
        this.setState(() => ({
            blockFilter: { x: null, y: null, z: null }
        }));
    }
}
