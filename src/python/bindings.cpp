// src/python/bindings.cpp
// Simplified Python bindings for game logic only
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>
#include <string>

#include "core/igamestate.h"
#include "core/game_export.h"

#include "games/chess/chess_state.h"
#include "games/go/go_state.h"
#include "games/gomoku/gomoku_state.h"

#include "utils/attack_defense_module.h"

namespace py = pybind11;

namespace alphazero {
namespace python {

// Convert C++ tensor representation to Python numpy arrays
py::array_t<float> tensorToNumpy(const std::vector<std::vector<std::vector<float>>>& tensor) {
    if (tensor.empty() || tensor[0].empty() || tensor[0][0].empty()) {
        return py::array_t<float>();
    }
    
    size_t channels = tensor.size();
    size_t height = tensor[0].size();
    size_t width = tensor[0][0].size();
    
    py::array_t<float> array({channels, height, width});
    py::buffer_info info = array.request();
    float* data = static_cast<float*>(info.ptr);
    
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                data[c * height * width + h * width + w] = tensor[c][h][w];
            }
        }
    }
    
    return array;
}

// Module definition
PYBIND11_MODULE(alphazero_py, m) {
    m.doc() = "AlphaZero Python bindings - Game Logic Only";
    
    // Game types
    py::enum_<core::GameType>(m, "GameType")
        .value("UNKNOWN", core::GameType::UNKNOWN)
        .value("CHESS", core::GameType::CHESS)
        .value("GO", core::GameType::GO)
        .value("GOMOKU", core::GameType::GOMOKU)
        .export_values();
    
    // Game result
    py::enum_<core::GameResult>(m, "GameResult")
        .value("ONGOING", core::GameResult::ONGOING)
        .value("WIN_PLAYER1", core::GameResult::WIN_PLAYER1)
        .value("WIN_PLAYER2", core::GameResult::WIN_PLAYER2)
        .value("DRAW", core::GameResult::DRAW)
        .export_values();
    
    // Game interface
    py::class_<core::IGameState>(m, "IGameState")
        .def("get_legal_moves", &core::IGameState::getLegalMoves)
        .def("is_legal_move", &core::IGameState::isLegalMove)
        .def("make_move", &core::IGameState::makeMove)
        .def("undo_move", &core::IGameState::undoMove)
        .def("is_terminal", &core::IGameState::isTerminal)
        .def("get_game_result", &core::IGameState::getGameResult)
        .def("get_current_player", &core::IGameState::getCurrentPlayer)
        .def("get_board_size", &core::IGameState::getBoardSize)
        .def("get_action_space_size", &core::IGameState::getActionSpaceSize)
        .def("get_tensor_representation", [](const core::IGameState& state) {
            return tensorToNumpy(state.getTensorRepresentation());
        })
        .def("get_enhanced_tensor_representation", [](const core::IGameState& state) {
            return tensorToNumpy(state.getEnhancedTensorRepresentation());
        })
        .def("get_hash", &core::IGameState::getHash)
        .def("action_to_string", &core::IGameState::actionToString)
        .def("string_to_action", &core::IGameState::stringToAction)
        .def("to_string", &core::IGameState::toString)
        .def("get_move_history", &core::IGameState::getMoveHistory)
        .def("clone", &core::IGameState::clone);
    
    // Game factory
    m.def("create_game", [](core::GameType type) {
        return core::GameFactory::createGame(type);
    });
    
    m.def("create_game_from_moves", [](core::GameType type, const std::string& moves) {
        return core::GameFactory::createGameFromMoves(type, moves);
    });
    
    // Game serialization
    m.def("save_game", &core::GameSerializer::saveGame);
    m.def("load_game", &core::GameSerializer::loadGame);
    m.def("serialize_game", &core::GameSerializer::serializeGame);
    m.def("deserialize_game", &core::GameSerializer::deserializeGame);
    
    // Utility functions
    m.def("game_type_to_string", &core::gameTypeToString);
    m.def("string_to_game_type", &core::stringToGameType);
    
    // Game-specific classes with all options exposed
    
    // Chess with chess960 support
    py::class_<games::chess::ChessState, core::IGameState>(m, "ChessState")
        .def(py::init<>())
        .def(py::init<bool>(), py::arg("chess960") = false)
        .def(py::init<bool, const std::string&>(), py::arg("chess960") = false, py::arg("fen") = "")
        .def(py::init<bool, const std::string&, int>(), 
             py::arg("chess960") = false, py::arg("fen") = "", py::arg("position_number") = -1)
        .def(py::init<const games::chess::ChessState&>());

    // Go with rule set options
    py::enum_<games::go::GoState::RuleSet>(m, "GoRuleSet")
        .value("CHINESE", games::go::GoState::RuleSet::CHINESE)
        .value("JAPANESE", games::go::GoState::RuleSet::JAPANESE)
        .value("KOREAN", games::go::GoState::RuleSet::KOREAN)
        .export_values();
        
    py::class_<games::go::GoState, core::IGameState>(m, "GoState")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("board_size") = 19)
        .def(py::init<int, float, bool, bool>(), 
             py::arg("board_size") = 19, py::arg("komi") = 7.5f, 
             py::arg("chinese_rules") = true, py::arg("enforce_superko") = true)
        .def(py::init<int, games::go::GoState::RuleSet, float>(),
             py::arg("board_size"), py::arg("rule_set"), py::arg("custom_komi") = -1.0f);

    // Gomoku with all rule variants
    py::class_<games::gomoku::GomokuState, core::IGameState>(m, "GomokuState")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("board_size") = 15)
        .def(py::init<int, bool, bool, int, bool>(),
             py::arg("board_size") = 15, py::arg("use_renju") = false, 
             py::arg("use_omok") = false, py::arg("seed") = 0,
             py::arg("use_pro_long_opening") = false)
        .def("get_renju_rules", &games::gomoku::GomokuState::getRenjuRules)
        .def("get_omok_rules", &games::gomoku::GomokuState::getOmokRules)
        .def("get_pro_long_opening", &games::gomoku::GomokuState::getProLongOpening);
    
    // Attack/Defense Module bindings
    py::class_<AttackDefenseModule>(m, "AttackDefenseModule");
    
    py::class_<GomokuAttackDefenseModule, AttackDefenseModule>(m, "GomokuAttackDefenseModule")
        .def(py::init<int>(), py::arg("board_size"))
        .def("compute_bonuses", [](GomokuAttackDefenseModule& self,
                                   const std::vector<std::vector<std::vector<int>>>& board_batch,
                                   const std::vector<int>& chosen_moves,
                                   const std::vector<int>& player_batch) {
            auto [attack_bonuses, defense_bonuses] = self.compute_bonuses(board_batch, chosen_moves, player_batch);
            return py::make_tuple(attack_bonuses, defense_bonuses);
        }, py::arg("board_batch"), py::arg("chosen_moves"), py::arg("player_batch"))
        .def("compute_planes", [](GomokuAttackDefenseModule&,
                                 const std::vector<core::IGameState*>& states) {
            // Note: Need to adapt this based on actual implementation
            // For now, return empty planes
            std::vector<std::vector<std::vector<float>>> attack_planes;
            std::vector<std::vector<std::vector<float>>> defense_planes;
            return py::make_tuple(attack_planes, defense_planes);
        }, py::arg("states"));
    
    // Helper function to compute attack/defense planes for a single state
    m.def("compute_attack_defense_planes", [](core::IGameState* state, const std::string& game_type) {
        // Get board size
        int board_size = state->getBoardSize();
        
        // Create appropriate module based on game type
        if (game_type == "gomoku") {
            GomokuAttackDefenseModule module(board_size);
            
            // Get board representation
            auto tensor = state->getTensorRepresentation();
            
            // Compute attack/defense scores for each position
            std::vector<std::vector<float>> attack_plane(board_size, std::vector<float>(board_size, 0.0f));
            std::vector<std::vector<float>> defense_plane(board_size, std::vector<float>(board_size, 0.0f));
            
            // Simple heuristic: Check for patterns around each empty position
            // This is a placeholder - real implementation would use the C++ module
            
            // Convert to numpy arrays
            py::array_t<float> attack_array({board_size, board_size});
            py::array_t<float> defense_array({board_size, board_size});
            
            auto attack_ptr = static_cast<float*>(attack_array.mutable_unchecked<2>().mutable_data(0, 0));
            auto defense_ptr = static_cast<float*>(defense_array.mutable_unchecked<2>().mutable_data(0, 0));
            
            for (int i = 0; i < board_size; ++i) {
                for (int j = 0; j < board_size; ++j) {
                    attack_ptr[i * board_size + j] = attack_plane[i][j];
                    defense_ptr[i * board_size + j] = defense_plane[i][j];
                }
            }
            
            return py::make_tuple(attack_array, defense_array);
        }
        
        // Return zeros for other game types
        py::array_t<float> zeros({board_size, board_size});
        return py::make_tuple(zeros, zeros);
    }, py::arg("state"), py::arg("game_type"));
}

} // namespace python
} // namespace alphazero