import React, {PureComponent} from 'react';
import {select} from 'd3';
import {zoom} from 'd3-zoom';
import * as d3 from 'd3';
import GridLayout from 'd3-v4-grid';
import {range} from 'd3-array';
import * as Konva from 'konva';
import {GridData, GridSelection} from './grid-data';

interface Props
{
    data: GridData<any>;
    selection: GridSelection;
    canvasDimensions: {width: number, height: number};
}

export class Grid extends PureComponent<Props>
{
    /*konva = (data: number[]) =>
    {
        const width = 1200;
        const height = 600;
        const stage = new Konva.Stage({
            container: this.props.id,
            width: width,
            height: height
        });
        const layer = new Konva.Layer();
        stage.add(layer);

        const grid = GridLayout() // create new grid layout
            .data(data)
            .padding([0.1, 0.1])
            .bands(true)
            .size([width, height]); // set size of container

        grid.layout();

        for (const d of grid.nodes())
        {
            layer.add(new Konva.Rect({
                x: d.x,
                y: d.y,
                width: grid.nodeSize()[0],
                height: grid.nodeSize()[1],
                fill: 'blue'
            }));
        }

        layer.draw();

        let mouseDown = false;

        stage.on('mousedown', () => mouseDown = true);
        stage.on('mouseup', () => mouseDown = false);
        // stage.on('mouseleave', () => mouseDown = false);
        stage.on('mousemove', (x: any) => {
            // console.log(x);
            // console.log(mouseDown);
            if (mouseDown)
            {
                stage.offset({
                    x: stage.offset().x + x.evt.movementX,
                    y: stage.offset().y + x.evt.movementY
                });
                stage.batchDraw();
            }
        });

        const scaleBy = 1.01;
        stage.on('wheel', (e: any) =>
        {
            e.evt.preventDefault();
            const oldScale = stage.scaleX();
            const mousePointTo = {
                x: stage.getPointerPosition().x / oldScale - stage.x() / oldScale,
                y: stage.getPointerPosition().y / oldScale - stage.y() / oldScale,
            };
            const newScale = e.evt.deltaY > 0 ? oldScale * scaleBy : oldScale / scaleBy;
            stage.scale({ x: newScale, y: newScale });
            const newPos = {
                x: -(mousePointTo.x - stage.getPointerPosition().x / newScale) * newScale,
                y: -(mousePointTo.y - stage.getPointerPosition().y / newScale) * newScale
            };
            stage.position(newPos);
            stage.batchDraw();
        });
    }

    componentDidMount()
    {
        this.d3(range(64 * 64));
    }

    d3 = (data: number[], selected: number = 0) =>
    {
        const container = select('#d3-test');
        //select('body')
        //    .on('keydown', x => {
        //        this.d3(data, selected + 1);
        //        console.log('keydown');
        //    });

        const width = 1200;//(container.node() as any).getBoundingClientRect().width;
        const height = 600;//(container.node() as any).getBoundingClientRect().height;

        const svg = container.append('svg')
            .attr('width', width)
            .attr('height', height);

        const grid = GridLayout() // create new grid layout
            .data(data)
            .padding([0.1, 0.1])
            .bands(true)
            .size([width, height]); // set size of container

        grid.layout();

        const nodeSize = grid.nodeSize();

        const g = svg.append('g');

        const z = zoom()
            .scaleExtent([1 / 2, 4])
            .translateExtent([
                [-nodeSize[0], -nodeSize[1]],
                [width + nodeSize[0], height + nodeSize[1]]]
            )
            .on('zoom', () => {
                g.attr('transform', d3.event.transform);
            });

        const selection = svg.append('rect')
            .attr('width', width)
            .attr('height', height)
            .style('fill', 'none')
            .style('pointer-events', 'all')
            .call(z);

        //z.scaleTo(selection as any, 8);

        const example = g
                .selectAll('rect')
                .data(grid.nodes())
                .attr('fill', (d, i: number) => i === selected ? 'red' : 'rgb(0,0,255)')
                .enter()
                .append('rect')
                .attr('x', (d: {x: number}) => d.x)
                .attr('y', (d: {y: number}) => d.y)
                .attr('width', nodeSize[0])
                .attr('height', nodeSize[1])
                .attr('fill', (d, i: number) => i === selected ? 'red' : 'rgb(0,0,255)');
    }

    render()
    {
        return <div id='d3-test'></div>;
    }*/

    render()
    {
        const layout = this.calculateLayout(
            { rows: this.props.selection.height + 1, cols: this.props.selection.width + 1 },
            { width: this.props.canvasDimensions.width, height: this.props.canvasDimensions.height }
        );
        const nodeSize = {
            width: layout.nodeSize()[0],
            height: layout.nodeSize()[1]
        };

        const axes = this.renderAxes(layout.nodes(), nodeSize, this.props.selection);
        const grid = this.renderGrid(layout.nodes(), nodeSize, this.props.selection);

        return (
            <svg width='100%' height='100%'
                 viewBox={`0 0 ${this.props.canvasDimensions.width} ${this.props.canvasDimensions.height}`}>
                <g>{axes}</g>
                <g>{grid}</g>
            </svg>
        );
    }

    renderAxes = (nodes: {x: number, y: number}[],
                  nodeSize: {width: number, height: number},
                  selection: GridSelection): JSX.Element[] =>
    {
        const axes: JSX.Element[] = [];
        const addLabel = (index: number, text: string) =>
        {
            axes.push(
                <text
                    key={index}
                    x={nodes[index].x + nodeSize.width / 2}
                    y={nodes[index].y + nodeSize.height / 2}
                    width={nodeSize.width}
                    height={nodeSize.height}
                    fill='rgb(0, 0, 0)'
                    textAnchor='middle'
                    fontSize={16}
                    dominantBaseline={'central'}
                >{text}</text>
            );
        };

        // horizontal labels
        for (const index of range(selection.width))
        {
            addLabel(index + 1, `${selection.x + index + 1}`);
        }

        // vertical label
        for (const index of range(selection.height))
        {
            addLabel((index + 1) * (selection.width + 1), `${selection.y + index + 1}`);
        }

        return axes;
    }

    renderGrid = (nodes: Array<{x: number, y: number}>,
                  nodeSize: {width: number, height: number},
                  {width, height, z, y: startY, x: startX}: GridSelection): JSX.Element[] =>
    {
        const grid: JSX.Element[] = [];

        for (let y = 1; y < height + 1; y++)
        {
            for (let x = 1; x < width + 1; x++)
            {
                const index = y * (width + 1) + x;
                const element = this.getElement(z, startX + (x - 1), startY + (y - 1));

                grid.push(
                    <rect
                        key={index}
                        x={nodes[index].x}
                        y={nodes[index].y}
                        width={nodeSize.width}
                        height={nodeSize.height}
                        fill={element === null ? 'rgb(255, 255, 255)' : 'rgb(255, 0, 0)'}
                        stroke='rgb(0, 0, 0)'
                        strokeWidth={1} />
                );
            }
        }

        return grid;
    }

    calculateLayout(gridSize: {rows: number, cols: number},
                    canvasSize: {width: number, height: number})
    {
        const layout = GridLayout()
            .data(range(gridSize.rows * gridSize.cols))
            .padding([0.1, 0.1])
            .bands(true)
            .rows(gridSize.rows)
            .cols(gridSize.cols)
            .size([canvasSize.width, canvasSize.height]);
        layout.layout();

        return layout;
    }

    getElement = (z: number, x: number, y: number): {} | null =>
    {
        const zDim = this.props.data[z];
        if (zDim === undefined) return null;
        const xDim = zDim[x];
        if (xDim === undefined) return null;
        const yDim = xDim[y];
        if (yDim === undefined) return null;

        return yDim;
    }
}
