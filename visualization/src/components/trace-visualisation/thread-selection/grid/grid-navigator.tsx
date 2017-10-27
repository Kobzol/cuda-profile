import React, {ChangeEvent, PureComponent} from 'react';
import {GridBounds, GridSelection} from './grid-data';
import * as _ from 'lodash';

interface Props
{
    dimensions: number[];
    selection: GridSelection;
    onSelectionChanged: (selection: GridSelection) => void;
    bounds: GridBounds;
}

export class GridNavigator extends PureComponent<Props>
{
    render()
    {
        return (
            <div>
                {this.renderNumberSelect(this.props.dimensions, this.props.selection.width, 'width')}
                {this.renderNumberSelect(this.props.dimensions, this.props.selection.height, 'height')}
                {this.renderNumberSelect(_.range(0, this.props.bounds.z), this.props.selection.z, 'z')}
                {this.renderMoveButton(this.props.bounds.x, this.props.selection.x, this.props.selection.width,
                    'r', false)}
                {this.renderMoveButton(this.props.bounds.x, this.props.selection.x, this.props.selection.width,
                    'l', true)}
                {this.renderMoveButton(this.props.bounds.y, this.props.selection.y, this.props.selection.height,
                    'u', false)}
                {this.renderMoveButton(this.props.bounds.y, this.props.selection.y, this.props.selection.height,
                    'd', true)}
            </div>
        );
    }

    renderNumberSelect = (dimensions: number[], value: number, attribute: string): JSX.Element =>
    {
        return (
            <select
                onChange={(event: ChangeEvent<HTMLSelectElement>) => {
                    this.changeSelection(attribute, parseInt(event.target.value, 10));
                }}
                value={value}>
                {dimensions.map(dim =>
                    <option
                        key={dim}
                        value={dim}>
                        {dim}
                    </option>
                )}
            </select>
        );
    }

    changeSelection = (attribute: string, value: number) =>
    {
        this.props.onSelectionChanged({
            ...this.props.selection,
            [attribute]: value
        });
    }

    private renderMoveButton(bound: number, selection: number,
                             dimension: number, name: string,
                             positive: boolean): JSX.Element
    {
        console.log(bound, selection, dimension);
        const enabled =
            (positive && (selection + dimension) < bound) ||
            (!positive && (selection > 0));

        return (
            <button disabled={!enabled}>{name}</button>
        );
    }
}
