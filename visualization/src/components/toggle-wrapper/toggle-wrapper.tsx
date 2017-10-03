import React, {PureComponent} from 'react';
import {Button, Glyphicon} from 'react-bootstrap';

interface Props
{
    showContent: boolean;
    onShow: () => void;
    toggleText: string;
}

export class ToggleWrapper extends PureComponent<Props>
{
    render()
    {
        const showContent = this.props.showContent;
        if (showContent)
        {
            return (
                <div>{this.props.children}</div>
            );
        }
        else return this.renderToggleButton();
    }

    renderToggleButton = (): JSX.Element =>
    {
        return (
            <Button
                onClick={this.props.onShow}
                bsStyle='primary'>
                <Glyphicon glyph='list' /> {this.props.toggleText}
            </Button>
        );
    }
}
