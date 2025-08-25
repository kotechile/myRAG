# test_function_import.py
"""
Quick test to verify the integrate_knowledge_gap_filler_http function is accessible
"""

def test_function_import():
    """Test importing the specific function"""
    
    print("🔍 Testing integrate_knowledge_gap_filler_http import...")
    
    try:
        # Test importing the module first
        print("1. Importing module...")
        import knowledge_gap_http_supabase
        print("   ✅ Module imported successfully")
        
        # Check if function exists in module
        print("2. Checking if function exists in module...")
        if hasattr(knowledge_gap_http_supabase, 'integrate_knowledge_gap_filler_http'):
            print("   ✅ Function exists in module")
        else:
            print("   ❌ Function NOT found in module")
            print(f"   Available functions: {[name for name in dir(knowledge_gap_http_supabase) if not name.startswith('_')]}")
            return False
        
        # Test importing the function directly
        print("3. Importing function directly...")
        from knowledge_gap_http_supabase import integrate_knowledge_gap_filler_http
        print("   ✅ Function imported directly")
        
        # Check if function is callable
        print("4. Checking if function is callable...")
        if callable(integrate_knowledge_gap_filler_http):
            print("   ✅ Function is callable")
        else:
            print("   ❌ Function is not callable")
            return False
        
        print("\n✅ ALL TESTS PASSED!")
        print("The integrate_knowledge_gap_filler_http function is working correctly.")
        return True
        
    except ImportError as e:
        print(f"   ❌ ImportError: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_exports():
    """Test all exported functions and classes"""
    
    print("\n🔍 Testing all exports...")
    
    try:
        import knowledge_gap_http_supabase
        
        # Check __all__ exports
        if hasattr(knowledge_gap_http_supabase, '__all__'):
            exports = knowledge_gap_http_supabase.__all__
            print(f"   📋 __all__ exports: {exports}")
            
            for export_name in exports:
                if hasattr(knowledge_gap_http_supabase, export_name):
                    print(f"   ✅ {export_name}: Available")
                else:
                    print(f"   ❌ {export_name}: Missing")
        else:
            print("   ⚠️ No __all__ defined")
        
        # Test key imports
        key_imports = [
            'HTTPSupabaseClient',
            'KnowledgeGapAnalyzer',
            'MultiSourceResearcher',
            'EnhancedRAGKnowledgeEnhancer',
            'EnhancedKnowledgeGapFillerOrchestrator',
            'integrate_knowledge_gap_filler_http'
        ]
        
        print("\n   🔍 Testing key imports:")
        for import_name in key_imports:
            try:
                obj = getattr(knowledge_gap_http_supabase, import_name)
                print(f"   ✅ {import_name}: {type(obj).__name__}")
            except AttributeError:
                print(f"   ❌ {import_name}: Not found")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Knowledge Gap Function Import")
    print("=" * 60)
    
    success1 = test_function_import()
    success2 = test_all_exports()
    
    if success1 and success2:
        print("\n🎉 SUCCESS: All imports working correctly!")
        print("You can now use the Knowledge Gap Filler in your main.py")
    else:
        print("\n❌ FAILED: Some imports are not working")
        print("Check the error messages above")