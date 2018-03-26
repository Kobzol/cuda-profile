import React, {PureComponent} from 'react';
import {Kernel} from '../../../lib/profile/kernel';
import {Trace} from '../../../lib/profile/trace';
import {AccessType, Warp} from '../../../lib/profile/warp';
import {WarpFilter} from './warp-filter';
import {Dim3} from '../../../lib/profile/dim3';
import {WarpOverview} from './warp-overview/warp-overview';
import {Button, ListGroup, ListGroupItem, Card, CardHeader, CardBody} from 'reactstrap';
import {SourceLocation} from '../../../lib/profile/metadata';
import {SourceModal} from './source-modal/source-modal';
import styled from 'styled-components';
import {TraceHeader} from './trace-header';
import {TraceSelection} from '../../../lib/trace/selection';
import {contains} from 'ramda';
import {getFilename} from '../../../lib/util/string';
import MdViewList from 'react-icons/lib/md/view-list';
import MdClose from 'react-icons/lib/md/close';
import {AccessFilter, AccessTypeFilter} from './access-type-filter';

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
    typeFilter: AccessFilter;
    sourceModalOpened: boolean;
    activePanels: number[];
}

const Wrapper = styled(Card)`
  
`;
const Header = styled(CardHeader)`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
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
const FilterWrapper = styled.div`
  margin-bottom: 10px;
`;
const Block = styled.div`
  margin-bottom: 10px;
`;
const Label = styled.span`
  margin-right: 5px;
`;
const SourceList = styled(ListGroup)`
  margin-bottom: 5px;
`;

export class WarpPanel extends PureComponent<Props, State>
{
    state: State = {
        blockFilter: { x: null, y: null, z: null },
        locationFilter: [],
        typeFilter: {
            read: true,
            write: true
        },
        sourceModalOpened: false,
        activePanels: []
    };

    render()
    {
        const warps = this.getFilteredWarps();
        return (
            <Wrapper>
                <Header>
                    <span>Selected kernel</span>
                    <Button
                        onClick={this.deselectTrace}
                        color='primary' outline title='Change kernel'>
                        <MdViewList />
                    </Button>
                </Header>
                <BodyWrapper>
                    <TraceHeader
                        kernel={this.props.kernel}
                        trace={this.props.trace} />
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
                        <h4>Filters</h4>
                        {this.renderFilters(warps)}
                    </Section>
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
    renderFilters = (warps: Warp[]): JSX.Element =>
    {
        const label = `${warps.length} accesses selected by filter (${this.props.trace.warps.length} total)`;
        const location = this.state.locationFilter.map(loc =>
            <SourceLocationEntry key={`${loc.file}:${loc.line}`}>
                {getFilename(loc.file)}:{loc.line}
            </SourceLocationEntry>
        );

        return (
            <>
                <FilterWrapper>
                    <Block>
                        <Label>Block</Label>
                        <WarpFilter
                            filter={this.state.blockFilter}
                            onFilterChange={this.changeBlockFilter} />
                    </Block>
                    <Block>
                        <Label>Type</Label>
                        <AccessTypeFilter
                            filter={this.state.typeFilter}
                            onChange={this.handleTypeFilterChange} />
                    </Block>
                    {this.props.kernel.metadata.source &&
                    <div>
                        <Label>Source locations</Label>
                        <SourceList>{location}</SourceList>
                        <Button size='sm' onClick={this.openSourceModal}>Filter by source location</Button>
                    </div>
                    }
                </FilterWrapper>
                <div>{label}</div>
                {this.isFilterActive() &&
                    <Button onClick={this.resetFilters} color='danger'>
                        <MdClose/> Reset filter
                    </Button>
                }
            </>
        );
    }

    getFilteredWarps = (): Warp[] =>
    {
        const {x, y, z} = this.state.blockFilter;
        return this.props.trace.warps.filter(warp => {
            if (x !== null && warp.blockIdx.x !== x) return false;
            if (y !== null && warp.blockIdx.y !== y) return false;
            if (z !== null && warp.blockIdx.z !== z) return false;
            if (this.state.locationFilter.length > 0 && !this.testLocationFilter(warp)) return false;
            if (!this.state.typeFilter.read && warp.accessType === AccessType.Read) return false;
            if (!this.state.typeFilter.write && warp.accessType === AccessType.Write) return false;
            return true;
        });
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
            locationFilter: [],
            typeFilter: {
                read: true,
                write: true
            }
        }));
    }

    isFilterActive = (): boolean =>
    {
        return !(
            this.state.blockFilter.x === null &&
            this.state.blockFilter.y === null &&
            this.state.blockFilter.z === null &&
            this.state.typeFilter.read &&
            this.state.typeFilter.write &&
            this.state.locationFilter.length === 0
        );
    }

    setLocationFilter = (locationFilter: SourceLocation[]) =>
    {
        this.setState(() => ({ locationFilter }));
    }
    testLocationFilter = (warp: Warp): boolean =>
    {
        const location: SourceLocation = { file: warp.location.file, line: warp.location.line };
        return contains(location, this.state.locationFilter);
    }
    handleTypeFilterChange = (typeFilter: AccessFilter) =>
    {
        this.setState(() => ({
            typeFilter
        }));
    }

    deselectTrace = () =>
    {
        this.props.selectTrace(null);
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
