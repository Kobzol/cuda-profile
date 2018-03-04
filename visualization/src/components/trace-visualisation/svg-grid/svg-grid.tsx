import React, {PureComponent, Fragment} from 'react';
import GridLayout from 'd3-v4-grid';
import styled from 'styled-components';
import {range} from 'ramda';

export interface BlockParams
{
    index: number;
    x: number;
    y: number;
    width: number;
    height: number;
}

interface Props
{
    width: number;
    height: number;
    rows: number;
    cols: number;
    renderItem(params: BlockParams): JSX.Element;
}

const SVG = styled.svg`
  margin: 0 auto;
`;

export class SVGGrid extends PureComponent<Props>
{
    render()
    {
        return (
            <SVG width={this.props.width}
                 height={this.props.height}
                 viewBox={`0 0 ${this.props.width} ${this.props.height}`}>
                <g>{this.renderGrid()}</g>
            </SVG>
        );
    }
    renderGrid = (): JSX.Element[] =>
    {
        const layout = this.calculateLayout({
            rows: this.props.rows,
            cols: this.props.cols
        }, {
            width: this.props.width,
            height: this.props.height
        });

        const grid: JSX.Element[] = [];
        const nodes = layout.nodes();
        const nodeSize = layout.nodeSize();

        const width = this.props.cols;
        const height = this.props.rows;

        for (let y = 0; y < height; y++)
        {
            for (let x = 0; x < width; x++)
            {
                const index = y * (width) + x;
                grid.push(
                    <Fragment key={index}>
                        {this.props.renderItem({
                            index,
                            x: nodes[index].x,
                            y: nodes[index].y,
                            width: nodeSize[0],
                            height: nodeSize[1]}
                        )}
                    </Fragment>
                );
            }
        }

        return grid;
    }

    calculateLayout = (gridSize: {rows: number, cols: number},
                       canvasSize: {width: number, height: number}) =>
    {
        const layout = GridLayout()
            .data(range(0, gridSize.rows * gridSize.cols))
            .bands(true)
            .rows(gridSize.rows)
            .cols(gridSize.cols)
            .size([canvasSize.width, canvasSize.height]);
        layout.layout();

        return layout;
    }
}
