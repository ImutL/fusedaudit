// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// Minimal smoke-test example for the FusedAudit pipeline.
/// This contract intentionally contains a reentrancy vulnerability:
/// the external call is performed before the balance state is updated.
contract VulnerableVault {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() external {
        uint256 amount = balances[msg.sender];
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
        balances[msg.sender] = 0;
    }
}
