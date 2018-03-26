import React, {PureComponent} from 'react';
import styled from 'styled-components';
import {Input, InputGroup, InputGroupAddon, InputGroupText} from 'reactstrap';

export interface AccessFilter
{
    read: boolean;
    write: boolean;
}

interface Props
{
    filter: AccessFilter;
    onChange(filter: AccessFilter): void;
}

const padding = '3px';

const Group = styled(InputGroup)`
  margin-bottom: 5px;
`;
const CheckboxWrapper = styled(InputGroupText)`
  display: flex;
  height: 100%;
  align-items: center;
  padding: ${padding};
`;
const Label = styled(InputGroupText)`
  padding: ${padding};
  background-color: #FFFFFF;
`;

export class AccessTypeFilter extends PureComponent<Props>
{
    render()
    {
        return (
            <>
                <Group>
                    <InputGroupAddon addonType='prepend'>
                        <CheckboxWrapper>
                            <Input addon type='checkbox' checked={this.props.filter.read}
                                      onChange={this.handleTypeReadChange} />
                        </CheckboxWrapper>
                    </InputGroupAddon>
                    <Label>read</Label>
                </Group>
                <InputGroup>
                    <InputGroupAddon addonType='prepend'>
                        <CheckboxWrapper>
                            <Input addon type='checkbox' checked={this.props.filter.write}
                                      onChange={this.handleTypeWriteChange} />
                        </CheckboxWrapper>
                    </InputGroupAddon>
                    <Label>write</Label>
                </InputGroup>
            </>
        );
    }

    handleTypeReadChange = (event: React.FormEvent<HTMLInputElement>) =>
    {
        this.props.onChange({
            ...this.props.filter,
            read: event.currentTarget.checked
        });
    }
    handleTypeWriteChange = (event: React.FormEvent<HTMLInputElement>) =>
    {
        this.props.onChange({
            ...this.props.filter,
            write: event.currentTarget.checked
        });
    }
}
