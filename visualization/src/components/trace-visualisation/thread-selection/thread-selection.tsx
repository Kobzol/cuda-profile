import React, {PureComponent} from 'react';
import {MemoryAccess, MemoryAccessGroup} from '../../../lib/profile/memory-access';
import {Grid} from './grid/grid';
import {
    GridData, GridSelection, GridBounds, createBlockSelector
} from './grid/grid-data';
import {GridNavigator} from './grid/grid-navigator';
import {Selector} from 'reselect';

interface Props
{
    accessGroup: MemoryAccessGroup;
    bounds: GridBounds;
}

interface State
{
    selection: GridSelection;
    calculateData: Selector<MemoryAccessGroup, GridData<MemoryAccess>>;
}

const dimensions = [8, 16, 32, 64];

export class ThreadSelection extends PureComponent<Props, State>
{
    constructor(props: Props)
    {
        super(props);

        this.state = {
            selection: {
                z: 0,
                y: 0,
                x: 0,
                width: 32,
                height: 32
            },
            calculateData: createBlockSelector()
        };
    }

    render()
    {
        const data = this.state.calculateData(this.props.accessGroup);
        const selection = this.state.selection;

        return (
            <div>
                <GridNavigator
                    selection={selection}
                    onSelectionChanged={this.handleSelectionChange}
                    dimensions={dimensions}
                    bounds={this.props.bounds} />
                <Grid
                    data={data}
                    selection={selection}
                    canvasDimensions={{ width: 1600, height: 900 }} />
            </div>
        );
    }

    handleSelectionChange = (selection: GridSelection) =>
    {
        this.setState(() => ({selection}));
    }
}
