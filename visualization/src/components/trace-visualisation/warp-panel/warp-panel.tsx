import React, {PureComponent} from 'react';
import {Kernel} from '../../../lib/profile/kernel';
import {Trace} from '../../../lib/profile/trace';
import {Warp} from '../../../lib/profile/warp';
import {WarpFilter} from './warp-filter';
import {Dim3} from '../../../lib/profile/dim3';
import {WarpOverview} from './warp-overview/warp-overview';
import {Button, ListGroup, ListGroupItem, Card, CardHeader, CardBody} from 'reactstrap';
import {SourceLocation} from '../../../lib/profile/metadata';
import _ from 'lodash';
import {SourceModal} from './source-modal/source-modal';
import styled from 'styled-components';
import {TraceHeader} from './trace-header';
import {TraceSelection} from '../../../lib/trace/selection';

interface Props
{
    kernel: Kernel;
    trace: Trace;
    selectedWarps: Warp[];
    selectWarps(warps: Warp[]): void;
    selectTrace(selection: TraceSelection): void;
}

interface State
{
    blockFilter: Dim3;
    locationFilter: SourceLocation[];
    sourceModalOpened: boolean;
    activePanels: number[];
}

const Wrapper = styled(Card)`
  
`;
const BodyWrapper = styled(CardBody)`
  padding: 10px;
`;
const SourceLocationEntry = styled(ListGroupItem)`
  padding: 5px;
  font-size: 14px;
`;
const Section = styled.div`
  margin-top: 10px;
  :first-child {
    margin-top: 0;
  }
  
  h4 {
    margin: 0;
  }
`;

export class WarpPanel extends PureComponent<Props, State>
{
    state: State = {
        blockFilter: { x: null, y: null, z: null },
        locationFilter: [],
        sourceModalOpened: false,
        activePanels: []
    };

    render()
    {
        const warps = this.getFilteredWarps();
        return (
            <Wrapper>
                <CardHeader>Selected kernel</CardHeader>
                <BodyWrapper>
                    <TraceHeader
                        kernel={this.props.kernel}
                        trace={this.props.trace}
                        selectTrace={this.props.selectTrace} />
                    {this.props.kernel.metadata.source &&
                        <SourceModal
                            opened={this.state.sourceModalOpened}
                            kernel={this.props.kernel}
                            trace={this.props.trace}
                            locationFilter={this.state.locationFilter}
                            setLocationFilter={this.setLocationFilter}
                            onClose={this.closeSourceModal}/>
                    }
                    <Section>
                        <h4>Active filters</h4>
                        {this.isFilterActive() ? this.renderFilter(warps) :
                            `No filters (${this.props.trace.warps.length} accesses total)`}
                    </Section>
                    <Section>
                        <h4>Filter by block index</h4>
                        <WarpFilter
                            filter={this.state.blockFilter}
                            onFilterChange={this.changeBlockFilter} />
                    </Section>
                    {this.props.kernel.metadata.source &&
                        <Section>
                            <Button onClick={this.openSourceModal}>Filter by source location</Button>
                        </Section>
                    }
                    <Section>
                        <h4>Filtered access minimap</h4>
                        <WarpOverview
                            warps={warps}
                            selectedWarps={this.props.selectedWarps}
                            onWarpSelect={this.props.selectWarps} />
                    </Section>
                </BodyWrapper>
            </Wrapper>
        );
    }
    renderFilter = (warps: Warp[]): JSX.Element =>
    {
        const label = `${warps.length} accesses selected by filter (${this.props.trace.warps.length} total)`;

        const {x, y, z} = this.state.blockFilter;
        const dim = `${z || 'z'}.${y || 'y'}.${x || 'x'}`;
        const location = this.state.locationFilter.map(loc =>
            <SourceLocationEntry key={`${loc.file}:${loc.line}`}>
                {loc.file}:{loc.line}
            </SourceLocationEntry>
        );

        return (
            <div>
                <div>Block index: {dim}</div>
                {this.state.locationFilter.length > 0 &&
                <div>
                    Source locations:
                    <ListGroup>{location}</ListGroup>
                </div>}
                <div>{label}</div>
                <Button onClick={this.resetFilters} color='danger'>Reset filter</Button>
            </div>
        );
    }

    getFilteredWarps = (): Warp[] =>
    {
        const {x, y, z} = this.state.blockFilter;
        if (!this.isFilterActive()) return this.props.trace.warps;

        return this.props.trace.warps.filter(warp => {
            if (x !== null && warp.blockIdx.x !== x) return false;
            if (y !== null && warp.blockIdx.y !== y) return false;
            if (z !== null && warp.blockIdx.z !== z) return false;
            if (this.state.locationFilter.length > 0 && !this.testLocationFilter(warp)) return false;
            return true;
        });
    }

    isFilterActive = (): boolean =>
    {
        return (
            this.state.blockFilter.x !== null ||
            this.state.blockFilter.y !== null ||
            this.state.blockFilter.z !== null ||
            this.state.locationFilter.length > 0
        );
    }

    changeBlockFilter = (blockFilter: Dim3) =>
    {
        this.setState(() => ({
            blockFilter
        }));
    }
    resetFilters = () =>
    {
        this.setState(() => ({
            blockFilter: { x: null, y: null, z: null },
            locationFilter: []
        }));
    }

    setLocationFilter = (locationFilter: SourceLocation[]) =>
    {
        this.setState(() => ({ locationFilter }));
    }
    testLocationFilter = (warp: Warp): boolean =>
    {
        const location: SourceLocation = { file: warp.location.file, line: warp.location.line };
        return _.some(this.state.locationFilter, location);
    }

    changeSourcePanelVisibility = (sourceModalOpened: boolean) =>
    {
        this.setState(() => ({ sourceModalOpened }));
    }
    closeSourceModal = () =>
    {
        this.changeSourcePanelVisibility(false);
    }
    openSourceModal = () =>
    {
        this.changeSourcePanelVisibility(true);
    }
}
