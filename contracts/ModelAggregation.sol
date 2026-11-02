// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Contract for federated learning model updates
contract ModelAggregation {
    // Struct for model update metadata
    struct ModelUpdate {
        address client; // Client's Ethereum address
        string weightsHash; // Hash of model weights
        uint256 timestamp; // Submission timestamp
    }

    // Maps round number to array of updates
    mapping(uint256 => ModelUpdate[]) public updatesByRound;
    // Tracks current round
    uint256 public currentRound;

    // Event for when a client submits an update
    event UpdateSubmitted(
        uint256 round,
        address indexed client,
        string weightsHash,
        uint256 timestamp
    );

    // Submit model weights hash for current round
    function submitUpdate(string memory weightsHash) public {
        // Add update to current round
        updatesByRound[currentRound].push(
            ModelUpdate({
                client: msg.sender, // Caller address
                weightsHash: weightsHash, // Weights hash
                timestamp: block.timestamp // Current timestamp
            })
        );
        // Emit update event
        emit UpdateSubmitted(
            currentRound,
            msg.sender,
            weightsHash,
            block.timestamp
        );
    }

    // Advance to next round
    function nextRound() public {
        currentRound++; // Increment round
    }

    // Get updates for a specific round
    function getUpdates(uint256 round) public view returns (ModelUpdate[] memory) {
        return updatesByRound[round]; // Return updates array
    }
}
