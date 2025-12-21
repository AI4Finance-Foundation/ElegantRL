#!/bin/bash

# Quick start script for Tic-Tac-Toe example

echo "=========================================="
echo "ElegantRL Tic-Tac-Toe Example"
echo "=========================================="
echo ""

# Check if in correct directory
if [ ! -f "game.py" ]; then
    echo "Error: Please run this script from the examples/tictactoe directory"
    echo "Usage: cd examples/tictactoe && bash run_example.sh"
    exit 1
fi

# Parse command line argument
MODE=${1:-train}

if [ "$MODE" = "train" ]; then
    echo "Mode: Training"
    echo ""
    echo "Training agent with self-play for 20 iterations..."
    echo "This will take approximately 10-30 minutes depending on your hardware."
    echo ""
    python train.py --iterations 20 --games 100 --updates 500 --opponent self --gpu 0

elif [ "$MODE" = "eval" ]; then
    echo "Mode: Evaluation"
    echo ""

    # Check if model exists
    if [ ! -f "checkpoints/best_model.pth" ]; then
        echo "Error: No trained model found at checkpoints/best_model.pth"
        echo "Please run training first: bash run_example.sh train"
        exit 1
    fi

    echo "Evaluating trained agent against random opponent (100 games)..."
    echo ""
    python evaluate.py --model checkpoints/best_model.pth --mode vs_random --num-games 100

elif [ "$MODE" = "play" ]; then
    echo "Mode: Play Against Agent"
    echo ""

    # Check if model exists
    if [ ! -f "checkpoints/best_model.pth" ]; then
        echo "Error: No trained model found at checkpoints/best_model.pth"
        echo "Please run training first: bash run_example.sh train"
        exit 1
    fi

    echo "Starting interactive game against trained agent..."
    echo ""
    python evaluate.py --model checkpoints/best_model.pth --mode vs_human --agent-player 0

elif [ "$MODE" = "watch" ]; then
    echo "Mode: Watch Self-Play"
    echo ""

    # Check if model exists
    if [ ! -f "checkpoints/best_model.pth" ]; then
        echo "Error: No trained model found at checkpoints/best_model.pth"
        echo "Please run training first: bash run_example.sh train"
        exit 1
    fi

    echo "Watching agent play against itself (5 games)..."
    echo ""
    python evaluate.py --model checkpoints/best_model.pth --mode watch --num-games 5

else
    echo "Unknown mode: $MODE"
    echo ""
    echo "Usage: bash run_example.sh [MODE]"
    echo ""
    echo "Available modes:"
    echo "  train  - Train agent with self-play (default)"
    echo "  eval   - Evaluate agent against random opponent"
    echo "  play   - Play interactively against trained agent"
    echo "  watch  - Watch agent play against itself"
    echo ""
    echo "Examples:"
    echo "  bash run_example.sh train"
    echo "  bash run_example.sh eval"
    echo "  bash run_example.sh play"
    echo "  bash run_example.sh watch"
    exit 1
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
